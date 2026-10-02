from __future__ import annotations

import asyncio
from typing import cast

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.peer import RpcForwarder
from sharedlocalllm_backend.server_engine import ServerEngine
from test_runtime import FakeStore, LaunchServerEngine, SuccessfulLoadInference, runtime_with


def distributed_runtime():
    runtime = runtime_with(SuccessfulLoadInference())
    runtime.models[0].update(mtp=True, sizeBytes=8 * 1024**3, layerCount=4)
    cast(FakeStore, runtime.store).values["peer"] = {
        "id": "peer-1",
        "capabilities": {
            "id": "peer-1", "name": "Peer", "online": True,
            "gpu": {"vramAvailableGb": 8}, "ramAvailableGb": 16,
        },
    }
    return runtime


CONFIG = {
    "contextSize": 4096,
    "gpuLayers": [{"nodeId": "local", "layers": 3}, {"nodeId": "peer-1", "layers": 1}],
}


def test_server_progress_preserves_metadata_and_reads_live_bytes(monkeypatch) -> None:
    runtime = distributed_runtime()

    class Forwarder(RpcForwarder):
        async def start(self) -> str:
            self.bytes_to_worker = 1024
            self.bytes_from_worker = 128
            return "127.0.0.1:5000"

        async def stop(self) -> None:
            return None

    class ObservedServer(LaunchServerEngine):
        async def start(self, **kwargs) -> None:
            snapshot = runtime.snapshot()["cluster"]
            assert snapshot["engine"] == "llama-server"
            assert snapshot["workerNodeId"] == "peer-1"
            assert snapshot["expectedBytes"] == 2 * 1024**3
            assert snapshot["stage"] == "loading_weights"
            assert snapshot["bytesToWorker"] == 1024
            forwarder = runtime._server_forwarder
            assert forwarder is not None
            forwarder.bytes_to_worker += 2048
            assert runtime.snapshot()["cluster"]["bytesToWorker"] == 3072
            await super().start(**kwargs)

    monkeypatch.setattr("sharedlocalllm_backend.runtime.RpcForwarder", Forwarder)
    runtime.server_engine = cast(ServerEngine, ObservedServer())
    cluster = asyncio.run(runtime.start_cluster("model", CONFIG))
    assert cluster["status"] == "running"
    assert cluster["workerNodeId"] == "peer-1"


@pytest.mark.parametrize("failure_stage", ["tunnel", "loading_weights"])
def test_server_preparation_failure_cleans_up_progress_and_forwarder(
    monkeypatch, failure_stage: str,
) -> None:
    runtime = distributed_runtime()
    stopped: list[bool] = []

    class Forwarder(RpcForwarder):
        async def start(self) -> str:
            if failure_stage == "tunnel":
                raise BackendError("rpc_tunnel_failed", "diagnostic tunnel failure")
            return "127.0.0.1:5000"

        async def stop(self) -> None:
            stopped.append(True)

    class FailingServer(LaunchServerEngine):
        async def start(self, **kwargs) -> None:
            raise BackendError("llama_server_failed", "diagnostic weight failure")

    monkeypatch.setattr("sharedlocalllm_backend.runtime.RpcForwarder", Forwarder)
    runtime.server_engine = cast(ServerEngine, FailingServer())
    with pytest.raises(BackendError):
        asyncio.run(runtime.start_cluster("model", CONFIG))
    assert runtime.cluster["status"] == "error"
    assert "diagnostic" in runtime.cluster["error"]
    assert runtime._server_forwarder is None
    assert stopped == [True]
    assert runtime._active_compute_operation is None
