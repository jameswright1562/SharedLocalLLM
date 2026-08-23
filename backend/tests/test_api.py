from __future__ import annotations

import asyncio

import httpx

from sharedlocalllm_backend.api import create_openai_app


class Store:
    def get(self, key: str, default=None):
        return "secret" if key == "apiKey" else default


class Runtime:
    def __init__(self) -> None:
        self.store = Store()
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
