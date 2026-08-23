from __future__ import annotations

import asyncio
import json
import logging
import socket
import time

import httpx
import pytest
import uvicorn

from sharedlocalllm_backend.api import (
    API_LOG_LOGGER,
    create_control_app,
    create_openai_app,
    format_api_request,
    prepare_logged_body,
)
from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.runtime import BackendRuntime


class Store:
    def __init__(self, auth_required: bool = True) -> None:
        self.values: dict = {
            "apiKey": "secret",
            "authRequired": auth_required,
            "apiPort": 0,
        }

    def get(self, key: str, default=None):
        return self.values.get(key, default)


class Runtime:
    def __init__(self, auth_required: bool = True) -> None:
        self.store = Store(auth_required)
        self.cluster = {"status": "running", "modelId": "loaded-model", "coordinatorNodeId": "local"}
        self.local_node = {"id": "local"}
        self.models = [
            {"id": "loaded-model", "name": "Orchid 9B", "quantization": "Q4_K_M"},
            {"id": "other-model", "name": "Zephyr 8x7B", "quantization": "Q8_0"},
        ]

    async def chat(self, *_args, **_kwargs):
        return {"content": "hello", "reasoning": "", "tokensPerSecond": 1.0}


def tool_spec() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "Bash", "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }]


def tool_call() -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": "Bash", "arguments": '{"command":"Get-Location"}'},
    }


def test_models_lists_only_the_active_model() -> None:
    async def request():
        transport = httpx.ASGITransport(app=create_openai_app(Runtime()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(
                "/v1/models", headers={"Authorization": "Bearer secret"}
            )

    response = asyncio.run(request())
    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["loaded-model"]
    assert response.json()["data"][0]["aliases"] == ["orchid 9b", "orchid-9b-q4_k_m"]


def test_chat_accepts_name_and_slug_aliases_for_the_active_model() -> None:
    async def request(model: str):
        transport = httpx.ASGITransport(app=create_openai_app(Runtime()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            )

    assert asyncio.run(request("Orchid 9B")).status_code == 200
    assert asyncio.run(request("orchid-9b-q4_k_m")).status_code == 200
    assert asyncio.run(request("Zephyr 8x7B")).status_code == 200


def test_chat_falls_back_to_the_loaded_model_for_any_other_model_field() -> None:
    async def request(model: str | None):
        transport = httpx.ASGITransport(app=create_openai_app(Runtime()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload: dict = {"messages": [{"role": "user", "content": "hi"}]}
            if model is not None:
                payload["model"] = model
            return await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json=payload,
            )

    for model in ("different-model", "totally-missing", "DUP MODEL", None, "active"):
        response = asyncio.run(request(model))
        assert response.status_code == 200, model
        assert response.json()["model"] == "loaded-model"


def test_chat_rejects_missing_and_wrong_keys_when_authentication_is_required() -> None:
    async def request(headers: dict[str, str]):
        transport = httpx.ASGITransport(app=create_openai_app(Runtime(auth_required=True)))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    missing = asyncio.run(request({}))
    wrong = asyncio.run(request({"Authorization": "Bearer nope"}))
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert wrong.json()["error"]["type"] == "api_unauthorized"


def test_chat_serves_requests_without_a_key_when_authentication_is_disabled() -> None:
    async def request():
        transport = httpx.ASGITransport(app=create_openai_app(Runtime(auth_required=False)))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            models = await client.get("/v1/models")
            chat = await client.post(
                "/v1/chat/completions",
                json={"model": "loaded-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        return health, models, chat

    health, models, chat = asyncio.run(request())
    assert health.status_code == 200
    assert models.status_code == 200
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "hello"


def test_chat_forwards_tools_and_returns_openai_tool_calls() -> None:
    class ToolRuntime(Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.received: dict = {}

        async def chat(self, messages, settings, images, **kwargs):
            self.received = {"messages": messages, **kwargs}
            return {
                "content": "",
                "message": {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
                "finishReason": "tool_calls",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

    async def request(runtime: ToolRuntime):
        transport = httpx.ASGITransport(app=create_openai_app(runtime))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={
                    "messages": [{"role": "user", "content": "Run Get-Location"}],
                    "tools": tool_spec(), "tool_choice": "auto",
                },
            )

    runtime = ToolRuntime()
    response = asyncio.run(request(runtime))
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [tool_call()]
    assert choice["finish_reason"] == "tool_calls"
    assert response.json()["usage"]["total_tokens"] == 15
    assert runtime.received["tools"] == tool_spec()
    assert runtime.received["tool_choice"] == "auto"


def test_local_stream_relays_native_tool_call_deltas_and_finish_reason() -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{
            "index": 0,
            "delta": {"tool_calls": [{"index": 0, **tool_call()}]},
            "finish_reason": None,
        }]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]

    class StreamingInference:
        async def chat_openai_stream(self, messages, settings, tools, tool_choice):
            assert tools == tool_spec()
            assert tool_choice == "auto"
            for chunk in chunks:
                yield chunk

    class StreamingRuntime(Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.inference = StreamingInference()

    async def request():
        transport = httpx.ASGITransport(app=create_openai_app(StreamingRuntime()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={
                    "stream": True, "messages": [{"role": "user", "content": "Run it"}],
                    "tools": tool_spec(), "tool_choice": "auto",
                },
            )

    response = asyncio.run(request())
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line]
    assert events[-1] == "[DONE]"
    payloads = [json.loads(line) for line in events[:-1]]
    assert payloads[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Bash"
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_remote_stream_synthesizes_tool_call_and_terminal_chunks() -> None:
    class RemoteRuntime(Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.cluster["coordinatorNodeId"] = "peer"

        async def chat(self, *_args, **_kwargs):
            return {
                "content": "",
                "message": {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
                "finishReason": "tool_calls",
            }

    async def request():
        transport = httpx.ASGITransport(app=create_openai_app(RemoteRuntime()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={"stream": True, "messages": [], "tools": tool_spec()},
            )

    response = asyncio.run(request())
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line]
    payloads = [json.loads(line) for line in events[:-1]]
    streamed_call = payloads[0]["choices"][0]["delta"]["tool_calls"][0]
    assert streamed_call["index"] == 0
    assert streamed_call["function"]["name"] == "Bash"
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1] == "[DONE]"


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_try_api_request_round_trips_through_the_real_http_server() -> None:
    async def scenario():
        server = uvicorn.Server(
            uvicorn.Config(
                create_openai_app(Runtime()), host="127.0.0.1", port=0,
                log_level="warning", lifespan="off",
            )
        )
        task = asyncio.create_task(server.serve())
        try:
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(0.05)
            port = int(server.servers[0].sockets[0].getsockname()[1])
            runtime = BackendRuntime.__new__(BackendRuntime)
            runtime.store = Store()
            runtime.store.values["apiPort"] = port
            started = time.perf_counter()
            result = await BackendRuntime.try_api_request(runtime)
            assert result["status"] == 200
            assert time.perf_counter() - started >= 0
            body = json.loads(result["body"])
            assert body["choices"][0]["message"]["content"] == "hello"
            assert result["durationMs"] >= 0
        finally:
            server.should_exit = True
            await task

    asyncio.run(scenario())


def test_try_api_request_reports_an_unreachable_api() -> None:
    runtime = BackendRuntime.__new__(BackendRuntime)
    runtime.store = Store()
    runtime.store.values["apiPort"] = _free_loopback_port()

    with pytest.raises(BackendError) as excinfo:
        asyncio.run(BackendRuntime.try_api_request(runtime))

    assert excinfo.value.code == "api_unavailable"
    assert "127.0.0.1" in excinfo.value.message


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_every_openai_api_request_is_logged_with_status_and_duration() -> None:
    logger = logging.getLogger(API_LOG_LOGGER)
    handler = RecordingHandler()
    logger.addHandler(handler)
    try:
        async def requests():
            transport = httpx.ASGITransport(app=create_openai_app(Runtime()))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                unauthorized = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                authorized = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer secret"}
                )
            return unauthorized.status_code, authorized.status_code

        statuses = asyncio.run(requests())
    finally:
        logger.removeHandler(handler)

    assert statuses == (401, 200)
    assert any("POST /v1/chat/completions -> 401" in line for line in handler.lines)
    assert any("GET /v1/models -> 200" in line for line in handler.lines)
    assert all(line.startswith("[api] ") for line in handler.lines)


def test_request_bodies_are_logged_with_the_request_line() -> None:
    logger = logging.getLogger(API_LOG_LOGGER)
    handler = RecordingHandler()
    logger.addHandler(handler)
    try:
        async def request():
            transport = httpx.ASGITransport(app=create_openai_app(Runtime()))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer secret"},
                    json={
                        "model": "active",
                        "messages": [{"role": "user", "content": "Explain layer splitting"}],
                        "max_tokens": 32,
                    },
                )

        asyncio.run(request())
    finally:
        logger.removeHandler(handler)

    body_lines = [line for line in handler.lines if "body=" in line]
    assert len(body_lines) == 1
    assert '"content":"Explain layer splitting"' in body_lines[0]
    assert "Bearer secret" not in body_lines[0]


def test_logged_bodies_redact_image_data_and_truncate_oversized_payloads() -> None:
    image = "data:image/png;base64," + "A" * 50_000
    prepared = prepare_logged_body(
        json.dumps({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": image}},
        ]}]}).encode()
    )
    assert "AAAA" not in prepared
    assert "data:image/png;base64," in prepared
    assert "[redacted 50000 chars]" in prepared

    prepared_long = prepare_logged_body(json.dumps({"prompt": "x" * 5_000}).encode())
    assert len(prepared_long) < 5_000
    assert prepared_long.startswith('{"prompt": "')
    assert "[truncated " in prepared_long
    assert "\n" not in prepared_long


def test_responses_are_logged_in_full_without_truncation() -> None:
    long_answer = "y" * 6_000

    class LongChatRuntime(Runtime):
        async def chat(self, *_args, **_kwargs):
            return {"content": long_answer, "reasoning": "", "tokensPerSecond": 1.0}

    logger = logging.getLogger(API_LOG_LOGGER)
    handler = RecordingHandler()
    logger.addHandler(handler)
    try:
        async def request():
            transport = httpx.ASGITransport(app=create_openai_app(LongChatRuntime()))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/v1/completions",
                    headers={"Authorization": "Bearer secret"},
                    json={"model": "active", "prompt": "hi", "max_tokens": 8},
                )

        response = asyncio.run(request())
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 200
    assert len(handler.lines) == 1
    line = handler.lines[0]
    assert f'response={{"id":"cmpl-sharedlocalllm","object":"text_completion","choices":[{{"index":0,"text":"{long_answer}"' in line
    assert "[truncated" not in line
    assert "\n" not in line


def test_internal_control_traffic_is_not_request_logged() -> None:
    logger = logging.getLogger(API_LOG_LOGGER)
    handler = RecordingHandler()
    logger.addHandler(handler)
    try:
        async def request():
            transport = httpx.ASGITransport(app=create_control_app(Runtime()))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/health")

        response = asyncio.run(request())
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 200
    assert handler.lines == []


def test_format_api_request_is_a_single_redacted_line() -> None:
    line = format_api_request("POST", "/v1/chat/completions", 200, 1234.5)
    assert line == "[api] POST /v1/chat/completions -> 200 in 1234 ms"
    assert "\n" not in line

    with_body = format_api_request("POST", "/v1/chat/completions", 200, 12.0, '{"model": "active"}')
    assert with_body == '[api] POST /v1/chat/completions -> 200 in 12 ms body={"model": "active"}'
    assert "\n" not in with_body
