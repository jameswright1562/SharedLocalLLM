from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .errors import BackendError
from .models import model_slug
from .openai_compat import (
    buffered_stream_choices,
    chunk_payload,
    completion_finish_reason,
    completion_message,
    completion_usage,
    request_tool_options,
)

CONTROL_PORT = 11436
API_LOG_LOGGER = "sharedlocalllm.api"
MAX_LOGGED_BODY_CHARS = 2000
_IMAGE_DATA_PATTERN = re.compile(r"data:image/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")

_api_logger = logging.getLogger(API_LOG_LOGGER)
if not _api_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _api_logger.addHandler(_handler)
    _api_logger.setLevel(logging.INFO)
_api_logger.propagate = False


def _clean_logged_text(text: str) -> str:
    def redact(match: re.Match[str]) -> str:
        prefix, data = match.group(0).split(",", 1)
        return f"{prefix},[redacted {len(data)} chars]"

    return _IMAGE_DATA_PATTERN.sub(redact, text.replace("\r", " ").replace("\n", " "))


def prepare_logged_body(raw: bytes) -> str:
    """Request body text for terminal logs: image payloads redacted and capped."""
    text = _clean_logged_text(raw.decode("utf-8", errors="replace"))
    if len(text) > MAX_LOGGED_BODY_CHARS:
        dropped = len(text) - MAX_LOGGED_BODY_CHARS
        text = f"{text[:MAX_LOGGED_BODY_CHARS]}...[truncated {dropped} chars]"
    return text


def prepare_logged_response(raw: bytes) -> str:
    """Response body for terminal logs: redacted and newline-free, never truncated."""
    return _clean_logged_text(raw.decode("utf-8", errors="replace"))


def format_api_request(
    method: str,
    target: str,
    status: int,
    duration_ms: float,
    body: str | None = None,
) -> str:
    """One terminal line per API request; keys stay in headers and are never logged."""
    line = f"[api] {method} {target} -> {status} in {duration_ms:.0f} ms"
    return f"{line} body={body}" if body else line


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

    @app.post("/_internal/stream/{command}")
    async def stream_command(command: str, request: Request) -> Any:
        payload = await request.json()
        payload = payload if isinstance(payload, dict) else {}
        if command != "send_chat_message":
            raise BackendError(
                "stream_unsupported", f"Streaming is not available for command: {command}"
            )

        async def events():
            try:
                async for event in runtime.chat_stream_events(
                    payload.get("messages", []), payload.get("settings", {}), payload.get("images", [])
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except BackendError as error:
                yield f"data: {json.dumps({'type': 'error', 'message': error.message})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


class ResponseLoggingMiddleware:
    """Pure ASGI middleware: one terminal line per API request and response.

    Streams are tee'd, so the logged response body is complete — never
    truncated — with image data redacted and newlines collapsed.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_chunks: list[bytes] = []
        response_chunks: list[bytes] = []
        status = 0

        async def receive_wrapper() -> dict:
            message = await receive()
            if message.get("type") == "http.request":
                request_chunks.append(message.get("body") or b"")
            return message

        async def send_wrapper(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message.get("status") or 0)
            elif message["type"] == "http.response.body":
                response_chunks.append(message.get("body") or b"")
            await send(message)

        started = time.perf_counter()
        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            parts: list[str] = []
            if any(request_chunks):
                parts.append(prepare_logged_body(b"".join(request_chunks)))
            if any(response_chunks):
                parts.append(f"response={prepare_logged_response(b''.join(response_chunks))}")
            _api_logger.info(
                format_api_request(
                    str(scope.get("method", "")), str(scope.get("path", "")),
                    status or 500, duration_ms, " ".join(parts) or None,
                )
            )


def create_openai_app(runtime: Any) -> FastAPI:
    app = FastAPI(title="SharedLocalLLM OpenAI API", docs_url=None, redoc_url=None)
    app.add_middleware(ResponseLoggingMiddleware)

    def active_model_id() -> str | None:
        return runtime.cluster.get("modelId")

    def model_aliases(model_id: str) -> list[str]:
        model = next((value for value in runtime.models if value["id"] == model_id), None)
        if not model:
            return []
        slug = model_slug(model["name"], model.get("quantization", ""))
        return sorted({model["name"].lower(), slug} - {model_id})

    def authorize(authorization: str | None) -> None:
        if not runtime.store.get("authRequired", True):
            return
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
        active = active_model_id()
        data = []
        if active and runtime.cluster.get("status") == "running":
            data.append({
                "id": active,
                "object": "model",
                "owned_by": "sharedlocalllm",
                "aliases": model_aliases(active),
            })
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, authorization: str | None = Header(default=None)) -> Any:
        authorize(authorization)
        body = await request.json()
        # A requested model that is not the loaded one falls back to the loaded
        # model instead of failing; the response reports the id that served it.
        active_model = active_model_id()
        settings = {
            "systemPrompt": "", "temperature": body.get("temperature", 0.7),
            "maxTokens": body.get("max_completion_tokens", body.get("max_tokens", 512)),
        }
        tools, tool_choice = request_tool_options(body)
        if body.get("stream") and runtime.cluster.get("coordinatorNodeId") == runtime.local_node["id"]:
            async def events():
                server_engine = getattr(runtime, "server_engine", None)
                engine = (
                    server_engine
                    if server_engine is not None and server_engine.active
                    else runtime.inference
                )
                async for native in engine.chat_openai_stream(
                    body.get("messages", []), settings, tools, tool_choice
                ):
                    chunk = chunk_payload(native.get("choices", []), active_model or "active")
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        response = await runtime.chat(
            body.get("messages", []), settings, [], proxy_peer=True,
            tools=tools, tool_choice=tool_choice,
        )
        if body.get("stream"):
            async def remote_events():
                for choices in buffered_stream_choices(response):
                    chunk = chunk_payload(choices, active_model or "active")
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(remote_events(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-sharedlocalllm", "object": "chat.completion",
            "model": active_model or "active",
            "choices": [{
                "index": 0, "message": completion_message(response),
                "finish_reason": completion_finish_reason(response),
            }],
            "usage": completion_usage(response),
        }

    @app.post("/v1/completions")
    async def completions(request: Request, authorization: str | None = Header(default=None)) -> Any:
        authorize(authorization)
        body = await request.json()
        active_model = active_model_id()
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
        previous = self.port
        try:
            await self.start(port)
        except Exception:
            if previous is not None and previous != port:
                await self.start(previous)
            raise

    def is_healthy(self) -> bool:
        return bool(
            self.server
            and self.server.started
            and self.task
            and not self.task.done()
        )

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
