from __future__ import annotations

import asyncio

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.runtime import BackendRuntime


class FakeStore:
    def __init__(self) -> None:
        self.values = {"peer": None, "benchmarks": []}
        self.recorded: list[dict] = []

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def append_benchmark(self, value: dict) -> None:
        self.recorded.append(value)


class FailingLoadInference:
    model_id = None

    async def load(self, *_args) -> None:
        raise RuntimeError("diagnostic load failure")

    async def unload(self) -> None:
        return None


class FailingBenchmarkInference:
    model_id = "model"

    async def benchmark(self):
        raise RuntimeError("diagnostic benchmark failure")


class FakePeer:
    remote_models: list[dict] = []


def runtime_with(inference) -> BackendRuntime:
    runtime = BackendRuntime.__new__(BackendRuntime)
    runtime.store = FakeStore()
    runtime.local_node = {
        "id": "local",
        "name": "Local",
        "online": True,
        "gpu": {"vramAvailableGb": 8},
        "ramAvailableGb": 16,
    }
    runtime.models = [
        {
            "id": "model",
            "name": "Model",
            "sizeBytes": 1024,
            "contextLength": 4096,
            "layerCount": 1,
            "remoteOnly": False,
        }
    ]
    runtime.model_paths = {"model": "model.gguf"}
    runtime.cluster = {"status": "idle"}
    runtime.inference = inference
    runtime.peer = FakePeer()
    return runtime


def test_failed_load_publishes_error_instead_of_staying_loading() -> None:
    runtime = runtime_with(FailingLoadInference())
    with pytest.raises(BackendError):
        asyncio.run(
            runtime.start_cluster(
                "model",
                {"contextSize": 4096, "gpuLayers": [{"nodeId": "local", "layers": 1}]},
            )
        )
    assert runtime.cluster["status"] == "error"
    assert runtime.local_node["clusterStatus"] == "error"


def test_failed_benchmark_is_persisted() -> None:
    runtime = runtime_with(FailingBenchmarkInference())
    result = asyncio.run(runtime.run_inference_benchmark("model"))
    assert result[0]["recommended"] is False
    assert result[0]["error"] == "diagnostic benchmark failure"
    assert runtime.store.recorded == result
