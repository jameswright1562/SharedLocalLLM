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

    async def request(op: str, data=None):
        if op == "heartbeat":
            return {}
        if op == "upload":
            return {"size": len(data["payload"]) // 2}
        if op == "download":
            return {"payload": "x" * data["size"]}
        raise AssertionError(op)

    peer.request = request  # type: ignore[method-assign]
    result = asyncio.run(peer.network_benchmark())
    assert result["downMbps"] > result["upMbps"]
    assert result["packetLossPercent"] == 0
