from __future__ import annotations

import asyncio
import json
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
    app = FastAPI(title="SharedLocalLLM OpenAI API", docs_url=None, redoc_url=None)

    def authorize(authorization: str | None) -> None:
        expected = f"Bearer {runtime.store.get('apiKey')}"
        if authorization != expected:
            raise BackendError("api_unauthorized", "A valid SharedLocalLLM bearer key is required.")

    @app.exception_handler(BackendError)
    async def api_error(_request: Request, error: BackendError):
        status = 401 if error.code == "api_unauthorized" else 400
        return JSONResponse({"error": {"message": error.message, "type": error.code}}, status_code=status)

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
        if body.get("stream") and runtime.cluster.get("coordinatorNodeId") == runtime.local_node["id"]:
            async def events():
                async for content in runtime.inference.chat_stream(body.get("messages", []), settings):
                    chunk = {
                        "id": "chatcmpl-sharedlocalllm", "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        response = await runtime.chat(body.get("messages", []), settings, [], proxy_peer=True)
        content = response["content"]
        if body.get("stream"):
            async def remote_events():
                chunk = {
                    "id": "chatcmpl-sharedlocalllm", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(remote_events(), media_type="text/event-stream")
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
    """Runs the user-facing API on the same asyncio loop as the control/peer runtime."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.server: uvicorn.Server | None = None
        self.task: asyncio.Task[None] | None = None
        self.port: int | None = None

    async def start(self, port: int) -> None:
        if self.task and not self.task.done() and self.port == port:
            return
        await self.stop()
        config = uvicorn.Config(
            create_openai_app(self.runtime), host="127.0.0.1", port=port,
            log_level="warning", access_log=False, lifespan="off",
        )
        self.server = uvicorn.Server(config)
        self.task = asyncio.create_task(self.server.serve())
        self.port = port
        for _ in range(100):
            if self.server.started:
                return
            if self.task.done():
                await self.task
                raise BackendError("api_start_failed", f"OpenAI API failed to bind port {port}.")
            await asyncio.sleep(0.05)
        raise BackendError("api_start_timeout", f"OpenAI API did not bind port {port} in time.")

    async def restart(self, port: int) -> None:
        await self.start(port)

    async def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
        if self.task:
            try:
                await asyncio.wait_for(self.task, 3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.task.cancel()
        self.server = None
        self.task = None
        self.port = None
