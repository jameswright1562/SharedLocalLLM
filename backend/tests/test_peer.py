from __future__ import annotations

import asyncio

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.peer import PEER_PORT, PeerManager, _parse_endpoint


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
