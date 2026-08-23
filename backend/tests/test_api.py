from __future__ import annotations

import asyncio
import json
import socket
import time

import httpx
import pytest
import uvicorn

from sharedlocalllm_backend.api import create_openai_app
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

    async def chat(self, *_args, **_kwargs):
        return {"content": "hello", "reasoning": "", "tokensPerSecond": 1.0}


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


def test_chat_rejects_a_model_that_is_not_loaded() -> None:
    async def request():
        transport = httpx.ASGITransport(app=create_openai_app(Runtime()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={
                    "model": "different-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    response = asyncio.run(request())
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "model_not_active"


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
