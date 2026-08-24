from __future__ import annotations

import asyncio
from typing import cast

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.peer import (
    PEER_PORT,
    PEER_REQUEST_TIMEOUT_SECONDS,
    PeerManager,
    _parse_endpoint,
)


def test_manual_peer_endpoint_defaults_to_peer_port() -> None:
    assert _parse_endpoint("10.10.10.2") == ("10.10.10.2", PEER_PORT)
    assert _parse_endpoint("10.10.10.2:50000") == ("10.10.10.2", 50000)


def test_manual_peer_endpoint_rejects_hostnames() -> None:
    with pytest.raises(BackendError):
        _parse_endpoint("example.com")


@pytest.mark.parametrize("value", ["10.10.10.2:notaport", "10.10.10.2:0", "10.10.10.2:70000"])
def test_manual_peer_endpoint_rejects_invalid_ports(value: str) -> None:
    with pytest.raises(BackendError):
        _parse_endpoint(value)


def test_network_benchmark_measures_upload_and_download_separately(monkeypatch) -> None:
    from sharedlocalllm_backend import peer as peer_module

    ticks = iter(float(value) for value in range(14))
    monkeypatch.setattr(peer_module.time, "perf_counter", lambda: next(ticks))
    class Runtime:
        local_node = {"adapter": {"name": "Ethernet"}}

    peer = PeerManager(Runtime())

    async def request(op: str, data: dict[str, object] | None = None):
        if op == "heartbeat":
            return {}
        assert data is not None
        if op == "upload":
            payload = data["payload"]
            assert isinstance(payload, str)
            return {"size": len(payload) // 2}
        if op == "download":
            size = data["size"]
            assert isinstance(size, int)
            return {"payload": "x" * size}
        raise AssertionError(op)

    peer.request = request  # type: ignore[method-assign]
    result = asyncio.run(peer.network_benchmark())
    assert result["downMbps"] > result["upMbps"]
    assert result["packetLossPercent"] == 0


def test_chat_dispatch_preserves_tool_options() -> None:
    class Runtime:
        local_node = {"id": "local"}

        def __init__(self) -> None:
            self.received: dict = {}

        async def chat(self, messages, settings, images, **kwargs):
            self.received = {
                "messages": messages, "settings": settings, "images": images, **kwargs,
            }
            return {"content": ""}

    runtime = Runtime()
    peer = PeerManager(runtime)
    tools = [{"type": "function", "function": {"name": "Bash"}}]
    asyncio.run(peer._dispatch("chat", {
        "messages": [{"role": "user", "content": "hi"}],
        "settings": {}, "images": [], "tools": tools, "toolChoice": "auto",
    }))

    assert runtime.received["tools"] == tools
    assert runtime.received["tool_choice"] == "auto"
    assert runtime.received["proxy_peer"] is False


@pytest.mark.parametrize("op", ["chat", "benchmark_inference", "start_cluster"])
def test_slow_peer_operations_have_no_response_deadline(op: str) -> None:
    class Store:
        def get(self, key: str):
            assert key == "peer"
            return {"address": "10.10.10.2:49158"}

    class Runtime:
        store = Store()

    peer = PeerManager(Runtime())
    captured: dict[str, int | None] = {}

    async def request_to(host, port, requested_op, data=None, timeout=PEER_REQUEST_TIMEOUT_SECONDS):
        captured[requested_op] = timeout
        return {}

    peer.request_to = request_to  # type: ignore[method-assign]
    asyncio.run(peer.request(op))

    assert captured[op] is None


def test_control_peer_operations_keep_a_response_deadline() -> None:
    class Store:
        def get(self, key: str):
            assert key == "peer"
            return {"address": "10.10.10.2:49158"}

    class Runtime:
        store = Store()

    peer = PeerManager(Runtime())
    captured: dict[str, int | None] = {}

    async def request_to(host, port, op, data=None, timeout=PEER_REQUEST_TIMEOUT_SECONDS):
        captured[op] = timeout
        return {}

    peer.request_to = request_to  # type: ignore[method-assign]
    asyncio.run(peer.request("heartbeat"))

    assert captured["heartbeat"] == PEER_REQUEST_TIMEOUT_SECONDS


def test_peer_stream_delivers_events_before_generation_finishes() -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        class Store:
            def __init__(self) -> None:
                self.address = ""

            def get(self, key: str):
                return {"address": self.address} if key == "peer" else None

            def log(self, *_args) -> None:
                return None

        class Runtime:
            local_node = {"id": "coordinator"}

            def __init__(self) -> None:
                self.store = Store()

            async def chat_openai_stream(self, *_args, **_kwargs):
                yield {"choices": [{"delta": {"content": "first"}}]}
                await release.wait()
                yield {"choices": [{"delta": {"content": "second"}}]}

            async def cancel_generation(self) -> None:
                return None

        serving_runtime = Runtime()
        serving_peer = PeerManager(serving_runtime)
        server = await asyncio.start_server(serving_peer._handle_connection, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])
        client_runtime = Runtime()
        client_runtime.store.address = f"127.0.0.1:{port}"
        client_peer = PeerManager(client_runtime)
        stream = client_peer.stream("chat_openai_stream", {
            "messages": [], "settings": {}, "tools": None, "toolChoice": None,
        })
        try:
            first = await asyncio.wait_for(anext(stream), 1)
            assert first["choices"][0]["delta"]["content"] == "first"
            release.set()
            remaining = [event async for event in stream]
            assert remaining[0]["choices"][0]["delta"]["content"] == "second"
        finally:
            await getattr(stream, "aclose")()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_closing_peer_stream_cancels_coordinator_generation() -> None:
    async def scenario() -> None:
        cancelled = asyncio.Event()
        global_cancelled = False

        class Store:
            def __init__(self) -> None:
                self.address = ""

            def get(self, key: str):
                return {"address": self.address} if key == "peer" else None

            def log(self, *_args) -> None:
                return None

        class Runtime:
            local_node = {"id": "coordinator"}

            def __init__(self) -> None:
                self.store = Store()

            async def chat_openai_stream(self, *_args, **_kwargs):
                try:
                    yield {"choices": [{"delta": {"content": "first"}}]}
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            async def cancel_generation(self) -> None:
                nonlocal global_cancelled
                global_cancelled = True

        serving_runtime = Runtime()
        serving_peer = PeerManager(serving_runtime)
        server = await asyncio.start_server(serving_peer._handle_connection, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])
        client_runtime = Runtime()
        client_runtime.store.address = f"127.0.0.1:{port}"
        stream = PeerManager(client_runtime).stream("chat_openai_stream", {
            "messages": [], "settings": {}, "tools": None, "toolChoice": None,
        })
        try:
            await asyncio.wait_for(anext(stream), 1)
            await getattr(stream, "aclose")()
            await asyncio.wait_for(cancelled.wait(), 1)
            assert global_cancelled is False
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_cancelling_stream_relay_closes_the_active_generator() -> None:
    async def scenario() -> None:
        from sharedlocalllm_backend.peer_stream import serve_events

        closed = asyncio.Event()
        reader = asyncio.StreamReader()

        class Writer:
            def write(self, _data: bytes) -> None:
                return None

            async def drain(self) -> None:
                return None

        async def events():
            try:
                await asyncio.Event().wait()
                yield {}
            finally:
                closed.set()

        task = asyncio.create_task(
            serve_events(reader, cast(asyncio.StreamWriter, Writer()), events())
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closed.wait(), 1)

    asyncio.run(scenario())
