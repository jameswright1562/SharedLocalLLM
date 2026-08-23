from __future__ import annotations

import asyncio

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.runtime import BackendRuntime


class FakeStore:
    def __init__(self) -> None:
        self.values: dict = {"peer": None, "benchmarks": []}
        self.recorded: list[dict] = []
        self.entries: list[tuple[str, str]] = []

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def update(self, **values) -> None:
        self.values.update(values)

    def log(self, level: str, event: str, message: str = "") -> None:
        self.entries.append((level, event))

    def append_benchmark(self, value: dict) -> None:
        self.recorded.append(value)

    def save_model_load_config(self, model_id: str, config: dict) -> None:
        configs = dict(self.values.get("modelLoadConfigs") or {})
        configs[model_id] = config
        self.values["modelLoadConfigs"] = configs

    def model_load_configs(self) -> dict[str, dict]:
        return dict(self.values.get("modelLoadConfigs") or {})

    def logs(self) -> list[str]:
        return [f"{level} {event} {message}".strip() for level, event, message in self.entries]


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


class SuccessfulLoadInference:
    model_id = None

    async def load(self, *_args) -> None:
        return None

    async def unload(self) -> None:
        return None


class TemporaryBenchmarkInference:
    model_id = None

    async def load(self, *_args) -> None:
        return None

    async def unload(self) -> None:
        return None

    async def benchmark(self):
        return (10.0, 5.0)


class RunningModelInference:
    """Pretends the model is loaded; reloading it for a benchmark would be a bug."""

    model_id = "model"

    def __init__(self) -> None:
        self.benchmarked = False

    async def load(self, *_args, **_kwargs) -> None:
        raise AssertionError("a running model must not reload for benchmarking")

    async def unload(self) -> None:
        return None

    async def benchmark(self):
        self.benchmarked = True
        return (900.0, 30.0)


class FakePeer:
    remote_models: list[dict] = []


class RecordingInference:
    model_id = None

    def __init__(self) -> None:
        self.cancelled = False
        self.chatted = False

    async def chat(self, messages, settings, images):
        self.chatted = True
        return {"content": "local"}

    async def unload(self) -> None:
        return None

    def cancel(self) -> None:
        self.cancelled = True


class ProxyingPeer:
    remote_models: list[dict] = []

    def __init__(self, response=None) -> None:
        self.response = response if response is not None else {}
        self.calls: list[tuple[str, dict | None]] = []

    async def request(self, op: str, data: dict | None = None):
        self.calls.append((op, data))
        return self.response


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
    runtime.peer_active_model_id = None
    runtime.network = None
    runtime._runtime = {"status": "ready"}
    return runtime


def peer_runtime(inference, peer) -> BackendRuntime:
    runtime = runtime_with(inference)
    runtime.peer = peer
    runtime.store.values["peer"] = {"id": "peer-1", "name": "Worker"}
    runtime.peer_active_model_id = "model"
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


def test_successful_load_saves_the_config_for_the_next_launch() -> None:
    runtime = runtime_with(SuccessfulLoadInference())

    asyncio.run(
        runtime.start_cluster(
            "model",
            {"contextSize": 2048, "gpuLayers": [{"nodeId": "local", "layers": 1}], "batchSize": 256},
        )
    )

    saved = runtime.store.model_load_configs()["model"]
    assert saved["contextSize"] == 2048
    assert saved["gpuLayers"] == [{"nodeId": "local", "layers": 1, "kind": "gpu"}]
    assert saved["batchSize"] == 256
    assert runtime.snapshot()["modelLoadConfigs"] == {"model": saved}


def test_failed_load_keeps_the_previously_saved_config() -> None:
    runtime = runtime_with(FailingLoadInference())
    previous = {"contextSize": 8192, "gpuLayers": []}
    runtime.store.values["modelLoadConfigs"] = {"model": previous}

    with pytest.raises(BackendError):
        asyncio.run(runtime.start_cluster("model", {"contextSize": 512}))

    assert runtime.store.model_load_configs() == {"model": previous}


def test_temporary_benchmark_load_does_not_replace_a_saved_config() -> None:
    runtime = runtime_with(TemporaryBenchmarkInference())
    saved = {"contextSize": 8192, "gpuLayers": []}
    runtime.store.values["modelLoadConfigs"] = {"model": saved}

    asyncio.run(runtime.run_inference_benchmark("model"))

    assert runtime.store.model_load_configs() == {"model": saved}


def test_failed_benchmark_is_persisted() -> None:
    runtime = runtime_with(FailingBenchmarkInference())
    result = asyncio.run(runtime.run_inference_benchmark("model"))
    assert result[0]["recommended"] is False
    assert result[0]["error"] == "diagnostic benchmark failure"
    assert runtime.store.recorded == result


def test_benchmark_reuses_the_running_instance_without_reloading() -> None:
    inference = RunningModelInference()
    runtime = runtime_with(inference)
    runtime.cluster = {"status": "running", "coordinatorNodeId": "local", "modelId": "model"}

    result = asyncio.run(runtime.run_inference_benchmark("model"))

    assert inference.benchmarked is True
    assert result[0]["recommended"] is True
    assert result[0]["loadTimeSeconds"] == 0.0
    assert runtime.cluster["status"] == "running"


def test_benchmark_of_a_peer_running_model_is_forwarded() -> None:
    class UnloadedInference:
        model_id = None

        async def load(self, *_args, **_kwargs) -> None:
            raise AssertionError("peer-loaded models must not load here")

        async def unload(self) -> None:
            return None

        async def benchmark(self):
            raise AssertionError("the benchmark should run on the peer")

    peer = ProxyingPeer(response=[{"modelName": "Model", "recommended": True}])
    runtime = peer_runtime(UnloadedInference(), peer)

    result = asyncio.run(runtime.run_inference_benchmark("model"))

    assert peer.calls[-1] == ("benchmark_inference", {"modelId": "model"})
    assert result == [{"modelName": "Model", "recommended": True}]


def test_chat_proxies_when_only_the_peer_runs_the_model() -> None:
    inference = RecordingInference()
    peer = ProxyingPeer(response={"content": "remote"})
    runtime = peer_runtime(inference, peer)

    result = asyncio.run(runtime.chat([{"role": "user", "content": "hi"}], {}, []))

    assert result["content"] == "remote"
    assert peer.calls[-1][0] == "chat"
    assert inference.chatted is False


def test_chat_streams_through_the_peer_when_the_peer_runs_the_model() -> None:
    inference = RecordingInference()
    peer = ProxyingPeer(response={"content": "remote", "tokensPerSecond": 12.5})
    runtime = peer_runtime(inference, peer)

    async def collect() -> list[dict]:
        return [event async for event in runtime.chat_stream_events([{"role": "user", "content": "hi"}], {}, [])]

    events = asyncio.run(collect())
    assert [event["type"] for event in events] == ["token", "stats", "done"]
    assert events[0]["content"] == "remote"
    assert peer.calls[-1][0] == "chat"


def test_local_cluster_beats_peer_heartbeat_for_routing() -> None:
    inference = RecordingInference()
    peer = ProxyingPeer()
    runtime = peer_runtime(inference, peer)
    runtime.cluster = {"status": "running", "coordinatorNodeId": "local", "modelId": "model"}

    asyncio.run(runtime.chat([{"role": "user", "content": "hi"}], {}, []))

    assert peer.calls == []
    assert inference.chatted is True


def test_stop_cluster_stops_a_peer_loaded_model_from_this_computer() -> None:
    inference = RecordingInference()
    peer = ProxyingPeer(response={"status": "idle"})
    runtime = peer_runtime(inference, peer)

    cluster = asyncio.run(runtime.stop_cluster())

    assert peer.calls[-1][0] == "stop_cluster"
    assert cluster["status"] == "ready"


def test_cancel_generation_is_forwarded_to_the_peer() -> None:
    inference = RecordingInference()
    peer = ProxyingPeer()
    runtime = peer_runtime(inference, peer)

    asyncio.run(runtime.cancel_generation())

    assert peer.calls[-1][0] == "cancel_generation"
    assert inference.cancelled is True


def test_benchmark_generation_honours_cancellation() -> None:
    import threading

    from sharedlocalllm_backend.inference import InferenceEngine

    class StubLlm:
        def __init__(self) -> None:
            self.reset_calls = 0

        def tokenize(self, data, add_bos=False):
            return [1, 2]

        def reset(self):
            self.reset_calls += 1

        def eval(self, tokens):
            return None

        def create_completion(self, *, prompt, max_tokens, temperature, logits_processor):
            for processor in logits_processor:
                processor(None, object())
            raise AssertionError("cancellation should abort before sampling")

    engine = InferenceEngine.__new__(InferenceEngine)
    stub = StubLlm()
    engine.llm = stub
    engine._sync_lock = threading.RLock()
    engine._cancel = threading.Event()
    engine.reasoning = False

    def eval_sets_cancel(tokens):
        engine._cancel.set()
        return None

    stub.eval = eval_sets_cancel  # type: ignore[method-assign]
    with pytest.raises(BackendError):
        asyncio.run(asyncio.to_thread(engine._benchmark_sync))
    assert stub.reset_calls >= 2
