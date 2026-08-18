from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .errors import BackendError

CONTROL_PORT = 11436


def create_control_app(runtime: Any) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="SharedLocalLLM Control", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.exception_handler(BackendError)
    async def backend_error(_request: Request, error: BackendError):
        return JSONResponse(error.to_dict(), status_code=400)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "backend": "python"}

    @app.post("/_internal/{command}")
    async def command(command: str, request: Request) -> Any:
        payload = await request.json()
        return await runtime.dispatch(command, payload if isinstance(payload, dict) else {})

    return app


def create_openai_app(runtime: Any) -> FastAPI:
    app = FastAPI(title="SharedLocalLLM OpenAI API")

    def authorize(authorization: str | None) -> None:
        expected = f"Bearer {runtime.store.get('apiKey')}"
        if authorization != expected:
            raise BackendError("api_unauthorized", "A valid SharedLocalLLM bearer key is required.")

    @app.exception_handler(BackendError)
    async def api_error(_request: Request, error: BackendError):
        return JSONResponse({"error": {"message": error.message, "type": error.code}}, status_code=400)

    @app.get("/health")
    async def health(authorization: str | None = Header(default=None)) -> dict[str, str]:
        authorize(authorization)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        data = [{"id": model["id"], "object": "model", "owned_by": "sharedlocalllm"} for model in runtime.models]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, authorization: str | None = Header(default=None)) -> Any:
        authorize(authorization)
        body = await request.json()
        settings = {
            "systemPrompt": "", "temperature": body.get("temperature", 0.7),
            "maxTokens": body.get("max_tokens", 512),
        }
        response = await runtime.chat(body.get("messages", []), settings, [], proxy_peer=True)
        content = response["content"]
        if body.get("stream"):
            async def events():
                chunk = {
                    "id": "chatcmpl-sharedlocalllm", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-sharedlocalllm", "object": "chat.completion",
            "model": body.get("model", runtime.cluster.get("modelId", "active")),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/completions")
    async def completions(request: Request, authorization: str | None = Header(default=None)) -> Any:
        authorize(authorization)
        body = await request.json()
        response = await runtime.chat(
            [{"role": "user", "content": str(body.get("prompt", ""))}],
            {"systemPrompt": "", "temperature": body.get("temperature", 0.7), "maxTokens": body.get("max_tokens", 512)},
            [],
        )
        return {
            "id": "cmpl-sharedlocalllm", "object": "text_completion",
            "choices": [{"index": 0, "text": response["content"], "finish_reason": "stop"}],
        }

    return app


class ApiServerManager:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    def start(self, port: int) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive() and self._port == port:
                return
            self._stop_locked()
            config = uvicorn.Config(create_openai_app(self.runtime), host="127.0.0.1", port=port, log_level="warning")
            self._server = uvicorn.Server(config)
            self._thread = threading.Thread(target=self._server.run, name="openai-api", daemon=True)
            self._port = port
            self._thread.start()

    def restart(self, port: int) -> None:
        self.start(port)

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None
        self._port = None
