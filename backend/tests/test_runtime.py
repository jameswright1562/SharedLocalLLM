from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.autotune import Autotuner, topology_fingerprint
from sharedlocalllm_backend.inference import InferenceEngine
from sharedlocalllm_backend.peer import PeerManager
from sharedlocalllm_backend.runtime import BackendRuntime
from sharedlocalllm_backend.server_engine import ServerEngine
from sharedlocalllm_backend.store import Store


class FakeStore:
    def __init__(self) -> None:
        self.values: dict = {"peer": None, "benchmarks": []}
        self.recorded: list[dict] = []
        self.entries: list[tuple[str, str, str]] = []

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def update(self, **values) -> None:
        self.values.update(values)

    def log(self, level: str, event: str, message: str = "") -> None:
        self.entries.append((level, event, message))

    def append_benchmark(self, value: dict) -> None:
        self.recorded.append(value)

    def save_model_load_config(self, model_id: str, config: dict) -> None:
        configs = dict(self.values.get("modelLoadConfigs") or {})
        configs[model_id] = config
        self.values["modelLoadConfigs"] = configs

    def model_load_configs(self) -> dict[str, dict]:
        return dict(self.values.get("modelLoadConfigs") or {})

    def model_tunes(self) -> dict[str, dict]:
        return dict(self.values.get("modelTunes") or {})

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


class InactiveServerEngine:
    active = False
    model_id = None

    def available(self) -> Path | None:
        return None

    async def stop(self) -> None:
        return None

    def cancel(self) -> None:
        return None


class ActiveServerEngine:
    active = True
    model_id = "model"

    def __init__(self) -> None:
        self.cancelled = False
        self.stopped = False
        self.received: dict = {}

    async def chat(self, messages, settings, tools=None, tool_choice=None):
        self.received = {
            "messages": messages, "settings": settings,
            "tools": tools, "tool_choice": tool_choice,
        }
        return {"content": "server", "message": {"role": "assistant", "content": "server"}}

    async def chat_stream(self, _messages, _settings):
        yield {"type": "token", "content": "server"}
        yield {"type": "done"}

    async def benchmark(self):
        return 100.0, 25.0

    async def stop(self) -> None:
        self.stopped = True
        self.active = False
        self.model_id = None

    def cancel(self) -> None:
        self.cancelled = True


class LaunchServerEngine(InactiveServerEngine):
    def __init__(self) -> None:
        self.active = False
        self.model_id = None
        self.started: dict = {}
        self.stopped = False

    def available(self):
        return Path("llama-server.exe")

    async def start(self, **kwargs) -> None:
        self.started = kwargs
        self.active = True
        self.model_id = kwargs["model_id"]

    async def stop(self) -> None:
        self.stopped = True
        self.active = False
        self.model_id = None


class RecordingInference:
    model_id = None

    def __init__(self) -> None:
        self.cancelled = False
        self.chatted = False

    async def chat(self, messages, settings, images, tools=None, tool_choice=None):
        self.chatted = True
        self.tools = tools
        self.tool_choice = tool_choice
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


def runtime_with(inference: object) -> BackendRuntime:
    runtime = BackendRuntime.__new__(BackendRuntime)
    runtime.store = cast(Store, FakeStore())
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
    runtime.inference = cast(InferenceEngine, inference)
    runtime.server_engine = cast(ServerEngine, InactiveServerEngine())
    runtime.autotuner = Autotuner(runtime.store)
    runtime._active_compute_operation = None
    runtime._server_forwarder = None
    runtime.peer = cast(PeerManager, FakePeer())
    runtime.peer_active_model_id = None
    runtime.network = None
    runtime._runtime = {"status": "ready"}
    return runtime


def peer_runtime(inference: object, peer: object) -> BackendRuntime:
    runtime = runtime_with(inference)
    runtime.peer = cast(PeerManager, peer)
    cast(FakeStore, runtime.store).values["peer"] = {"id": "peer-1", "name": "Worker"}
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


def test_cluster_launch_and_benchmark_are_blocked_while_autotuning() -> None:
    runtime = runtime_with(SuccessfulLoadInference())
    runtime._active_compute_operation = "model tuning"

    with pytest.raises(BackendError) as launch_error:
        asyncio.run(runtime.start_cluster("model", {"contextSize": 4096, "gpuLayers": []}))
    with pytest.raises(BackendError) as benchmark_error:
        asyncio.run(runtime.run_inference_benchmark("model"))

    assert launch_error.value.code == "compute_busy"
    assert benchmark_error.value.code == "compute_busy"


def test_autotune_is_blocked_while_a_benchmark_is_running() -> None:
    runtime = runtime_with(SuccessfulLoadInference())
    runtime._active_compute_operation = "inference benchmark"

    with pytest.raises(BackendError) as caught:
        asyncio.run(runtime.start_model_autotune("model"))

    assert caught.value.code == "compute_busy"


def test_stale_model_tune_is_rejected_without_overwriting_saved_config() -> None:
    runtime = runtime_with(SuccessfulLoadInference())
    saved = {"contextSize": 8192, "gpuLayers": []}
    cast(FakeStore, runtime.store).values["modelLoadConfigs"] = {"model": saved}
    cast(FakeStore, runtime.store).values["modelTunes"] = {
        "model": {
            "fingerprint": "old-topology",
            "winners": {
                "batchSize": 2048,
                "gpuLayersAllocations": [
                    {"nodeId": "removed-peer", "layers": 1, "kind": "gpu"},
                ],
            },
        },
    }
    assert topology_fingerprint(runtime._cluster_nodes()) != "old-topology"

    with pytest.raises(BackendError) as caught:
        runtime.apply_model_tune("model")

    assert caught.value.code == "tune_topology_changed"
    assert runtime.store.model_load_configs()["model"] == saved


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
    cast(FakeStore, runtime.store).values["modelLoadConfigs"] = {"model": previous}

    with pytest.raises(BackendError):
        asyncio.run(runtime.start_cluster("model", {"contextSize": 512}))

    assert runtime.store.model_load_configs() == {"model": previous}


def test_temporary_benchmark_load_does_not_replace_a_saved_config() -> None:
    runtime = runtime_with(TemporaryBenchmarkInference())
    saved = {"contextSize": 8192, "gpuLayers": []}
    cast(FakeStore, runtime.store).values["modelLoadConfigs"] = {"model": saved}

    asyncio.run(runtime.run_inference_benchmark("model"))

    assert runtime.store.model_load_configs() == {"model": saved}


def test_failed_benchmark_is_persisted() -> None:
    runtime = runtime_with(FailingBenchmarkInference())
    result = asyncio.run(runtime.run_inference_benchmark("model"))
    assert result[0]["recommended"] is False
    assert result[0]["error"] == "diagnostic benchmark failure"
    assert cast(FakeStore, runtime.store).recorded == result


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


def test_chat_forwards_tools_to_the_remote_coordinator() -> None:
    inference = RecordingInference()
    response = {
        "content": "", "message": {"role": "assistant", "content": None, "tool_calls": []},
        "finishReason": "tool_calls",
    }
    peer = ProxyingPeer(response=response)
    runtime = peer_runtime(inference, peer)
    tools = [{"type": "function", "function": {"name": "Bash"}}]

    result = asyncio.run(
        runtime.chat(
            [{"role": "user", "content": "hi"}], {}, [],
            tools=tools, tool_choice="auto",
        )
    )

    assert result == response
    assert peer.calls[-1] == (
        "chat",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "settings": {}, "images": [], "tools": tools, "toolChoice": "auto",
        },
    )


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


def test_active_server_engine_handles_agent_chat_stream_cancel_and_benchmark() -> None:
    inference = RecordingInference()
    server = ActiveServerEngine()
    runtime = runtime_with(inference)
    runtime.server_engine = cast(ServerEngine, server)
    runtime.cluster = {
        "status": "running", "coordinatorNodeId": "local",
        "modelId": "model", "engine": "llama-server",
    }
    tools = [{"type": "function", "function": {"name": "Bash"}}]

    result = asyncio.run(runtime.chat(
        [{"role": "user", "content": "pwd"}], {}, [],
        tools=tools, tool_choice="required",
    ))

    async def collect() -> list[dict]:
        return [event async for event in runtime.chat_stream_events([], {}, [])]

    events = asyncio.run(collect())
    asyncio.run(runtime.cancel_generation())
    benchmark = asyncio.run(runtime.run_inference_benchmark("model"))

    assert result["content"] == "server"
    assert server.received["tools"] == tools
    assert server.received["tool_choice"] == "required"
    assert events == [{"type": "token", "content": "server"}, {"type": "done"}]
    assert server.cancelled is True
    assert benchmark[0]["generationTokensPerSecond"] == 25.0
    assert inference.chatted is False


def test_mtp_model_launches_llama_server_instead_of_builtin_engine() -> None:
    class NoBuiltinLoad(SuccessfulLoadInference):
        async def load(self, *_args) -> None:
            raise AssertionError("an MTP model with llama-server installed must use server mode")

    runtime = runtime_with(NoBuiltinLoad())
    runtime.models[0]["mtp"] = True
    server = LaunchServerEngine()
    runtime.server_engine = cast(ServerEngine, server)

    cluster = asyncio.run(runtime.start_cluster(
        "model",
        {"contextSize": 4096, "gpuLayers": [{"nodeId": "local", "layers": 1}]},
    ))

    assert cluster["engine"] == "llama-server"
    assert server.started["model_path"] == "model.gguf"
    assert server.started["mtp"] is True
    assert server.started["rpc_endpoint"] is None


def test_server_engine_uses_and_stops_the_peer_rpc_forwarder(monkeypatch) -> None:
    class Forwarder:
        instances: list["Forwarder"] = []

        def __init__(self, peer, model_id, include_cpu=False) -> None:
            self.peer = peer
            self.model_id = model_id
            self.include_cpu = include_cpu
            self.stopped = False
            self.instances.append(self)

        async def start(self) -> str:
            return "127.0.0.1:5000"

        async def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr("sharedlocalllm_backend.runtime.RpcForwarder", Forwarder)
    runtime = runtime_with(SuccessfulLoadInference())
    runtime.models[0]["mtp"] = True
    runtime.models[0]["layerCount"] = 2
    server = LaunchServerEngine()
    runtime.server_engine = cast(ServerEngine, server)
    cast(FakeStore, runtime.store).values["peer"] = {
        "id": "peer-1",
        "capabilities": {
            "id": "peer-1", "name": "Peer", "online": True,
            "gpu": {"vramAvailableGb": 8}, "ramAvailableGb": 16,
        },
    }

    cluster = asyncio.run(runtime.start_cluster(
        "model",
        {
            "contextSize": 4096,
            "gpuLayers": [
                {"nodeId": "local", "layers": 1},
                {"nodeId": "peer-1", "layers": 1},
            ],
        },
    ))

    assert cluster["engine"] == "llama-server"
    assert server.started["rpc_endpoint"] == "127.0.0.1:5000"
    assert len(Forwarder.instances) == 1
    asyncio.run(runtime.stop_cluster())
    assert Forwarder.instances[0].stopped is True
    assert server.stopped is True


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


def test_peer_refresh_loop_survives_unexpected_errors(monkeypatch) -> None:
    from sharedlocalllm_backend import runtime as runtime_module

    runtime = runtime_with(SuccessfulLoadInference())
    calls = {"count": 0}
    failures = iter([KeyError("capabilities"), ValueError("bad port")])

    async def flaky_refresh() -> None:
        calls["count"] += 1
        failure = next(failures, None)
        if failure is not None:
            raise failure

    spins = {"count": 0}

    async def fast_sleep(_seconds: float) -> None:
        spins["count"] += 1
        if spins["count"] >= 6:
            raise asyncio.CancelledError

    monkeypatch.setattr(runtime_module.asyncio, "sleep", fast_sleep)
    runtime.refresh_peer = flaky_refresh  # type: ignore[method-assign]

    async def run() -> None:
        task = asyncio.create_task(runtime._peer_refresh_loop())
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())

    assert calls["count"] == 5
    warnings = [entry for entry in runtime.store.entries if entry[0] == "WARN"]
    assert len(warnings) == 2
