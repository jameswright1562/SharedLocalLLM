from __future__ import annotations

import ctypes
import threading
from pathlib import Path

import pytest

from sharedlocalllm_backend.rpc_native import (
    NativeRpcServer,
    classify_devices,
    prepare_model_devices,
    server_start_arguments,
    split_rpc_devices,
    tensor_split_for,
)


def device(name: str, type_: int = 1, registry: str = "CUDA", pointer: int = 0) -> dict:
    return {"name": name, "type": type_, "registry": registry, "pointer": pointer}


@pytest.fixture(autouse=True)
def clear_pending_devices():
    from sharedlocalllm_backend import rpc_native

    rpc_native._pending_devices[:] = []
    rpc_native._rpc_devices_by_endpoint.clear()
    yield
    rpc_native._pending_devices[:] = []
    rpc_native._rpc_devices_by_endpoint.clear()


def test_classify_devices_separates_rpc_servers_from_local_gpus() -> None:
    info = [
        device("CUDA0", type_=1, registry="CUDA", pointer=10),
        device("CPU", type_=0, registry="CPU", pointer=11),
        device("RPC0", type_=1, registry="RPC", pointer=12),
    ]
    rpc, gpus = classify_devices(info)
    assert [value["pointer"] for value in rpc] == [12]
    assert [value["pointer"] for value in gpus] == [10]


def test_classify_devices_keeps_registry_order() -> None:
    info = [
        device("RPC0", type_=1, registry="RPC", pointer=30),
        device("CUDA0", type_=1, registry="CUDA", pointer=31),
        device("RPC1", type_=1, registry="RPC", pointer=32),
    ]
    rpc, gpus = classify_devices(info)
    assert [value["pointer"] for value in rpc] == [30, 32]
    assert [value["pointer"] for value in gpus] == [31]


def test_classify_devices_falls_back_to_integrated_gpu() -> None:
    info = [
        device("IGPU0", type_=2, registry="Vulkan", pointer=20),
        device("RPC0", type_=1, registry="RPC", pointer=21),
    ]
    rpc, gpus = classify_devices(info)
    assert [value["pointer"] for value in gpus] == [20]


def test_tensor_split_orders_remote_before_local() -> None:
    assert tensor_split_for(1, 0, 1, 16, 0, 16) == [16.0, 16.0]


def test_tensor_split_spreads_across_multiple_devices() -> None:
    assert tensor_split_for(2, 0, 1, 12, 0, 6) == [6.0, 6.0, 6.0]
    assert tensor_split_for(0, 0, 2, 0, 0, 10) == [5.0, 5.0]
    assert tensor_split_for(1, 0, 0, 8, 0, 0) == [8.0]


def test_tensor_split_reserves_remote_cpu_between_gpu_and_local() -> None:
    assert tensor_split_for(1, 1, 1, 12, 4, 16) == [12.0, 4.0, 16.0]
    assert tensor_split_for(0, 1, 0, 0, 8, 0) == [8.0]


def test_split_rpc_devices_identifies_cpu_as_last_device() -> None:
    rpc = [device("RPC0", registry="RPC", pointer=30), device("RPC1", registry="RPC", pointer=31)]
    gpus, cpus = split_rpc_devices(rpc, include_cpu=True)
    assert [value["pointer"] for value in gpus] == [30]
    assert [value["pointer"] for value in cpus] == [31]


def test_split_rpc_devices_treats_all_as_gpus_without_cpu() -> None:
    rpc = [device("RPC0", registry="RPC", pointer=30), device("RPC1", registry="RPC", pointer=31)]
    gpus, cpus = split_rpc_devices(rpc, include_cpu=False)
    assert [value["pointer"] for value in gpus] == [30, 31]
    assert cpus == []


def test_pending_devices_are_injected_into_model_params() -> None:
    from sharedlocalllm_backend.rpc_native import _install_model_devices_patch

    llama_cpp = pytest.importorskip("llama_cpp")
    internals = pytest.importorskip("llama_cpp._internals")

    _install_model_devices_patch()
    prepare_model_devices([123, 456])
    params = llama_cpp.llama_model_default_params()
    with pytest.raises(ValueError):
        internals.LlamaModel(path_model="does-not-exist.gguf", params=params, verbose=False)
    array = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
    assert array[0] == 123
    assert array[1] == 456
    assert array[2] is None


def test_rpc_cache_dir_resolves_and_creates_the_llama_cpp_default(monkeypatch, tmp_path) -> None:
    from sharedlocalllm_backend.rpc_native import _rpc_cache_dir

    monkeypatch.setenv("LLAMA_CACHE", str(tmp_path))
    result = _rpc_cache_dir()
    assert result is not None
    assert (tmp_path / "llama.cpp" / "rpc").is_dir()
    assert Path(result).resolve() == (tmp_path / "llama.cpp" / "rpc").resolve()
    assert result.endswith(("/", "\\"))


def test_rpc_cache_dir_is_disabled_without_any_base_directory(monkeypatch, tmp_path) -> None:
    from sharedlocalllm_backend import rpc_native

    monkeypatch.setenv("LLAMA_CACHE", str(tmp_path))
    for key in ("LOCALAPPDATA", "XDG_CACHE_HOME", "HOME"):
        monkeypatch.delenv(key, raising=False)
    assert rpc_native._rpc_cache_root() == str(tmp_path)

    monkeypatch.delenv("LLAMA_CACHE", raising=False)
    assert rpc_native._rpc_cache_dir() is None


def test_server_start_arguments_pass_the_cache_directory() -> None:
    array = (ctypes.c_void_p * 1)(ctypes.c_void_p(7))
    endpoint, cache, threads, count, devices = server_start_arguments(
        "127.0.0.1:5000", "C:\\cache\\llama.cpp\\rpc\\", 8, 1, array
    )
    assert endpoint == b"127.0.0.1:5000"
    assert cache == b"C:\\cache\\llama.cpp\\rpc\\"
    assert threads == 8
    assert count == 1
    assert devices is array


def test_server_start_arguments_allow_a_disabled_cache() -> None:
    array = (ctypes.c_void_p * 1)(ctypes.c_void_p(7))
    _, cache, *_ = server_start_arguments("127.0.0.1:5000", None, 8, 1, array)
    assert cache is None


def test_prepare_rpc_load_distributed_builds_remote_first_split(monkeypatch, tmp_path) -> None:
    import asyncio

    llama_cpp = pytest.importorskip("llama_cpp")
    if getattr(llama_cpp, "__llama_stub__", False):
        pytest.skip("prepare_rpc_load needs the real llama.cpp native registry")

    from sharedlocalllm_backend.rpc_native import NativeRpcServer, prepare_rpc_load

    monkeypatch.setenv("LLAMA_CACHE", str(tmp_path))

    async def scenario() -> None:
        server = NativeRpcServer()
        endpoint = await asyncio.to_thread(server.start)
        split = await asyncio.to_thread(prepare_rpc_load, endpoint, 16, 0, 16)
        assert split is not None
        assert split[0] == 16.0
        assert abs(sum(split) - 32.0) < 0.001
        assert server.cache_dir is not None
        assert Path(server.cache_dir.rstrip("/\\")).is_dir()

    asyncio.run(scenario())


def test_prepare_rpc_load_reserves_remote_cpu_device(monkeypatch, tmp_path) -> None:
    import asyncio

    llama_cpp = pytest.importorskip("llama_cpp")
    if getattr(llama_cpp, "__llama_stub__", False):
        pytest.skip("prepare_rpc_load needs the real llama.cpp native registry")

    from sharedlocalllm_backend.rpc_native import NativeRpcServer, prepare_rpc_load

    monkeypatch.setenv("LLAMA_CACHE", str(tmp_path))

    async def scenario() -> None:
        server = NativeRpcServer()
        endpoint = await asyncio.to_thread(server.start, True)
        split = await asyncio.to_thread(prepare_rpc_load, endpoint, 12, 4, 16)
        assert split is not None
        # Local GPU is always last; the remote CPU share sits immediately before
        # it. Stale RPC devices from earlier loads/tests only add extra remote
        # GPU-class entries, so assert the ordering invariant rather than the
        # exact device count.
        assert split[-1] == 16.0
        assert split[-2] == 4.0
        assert abs(sum(split) - 32.0) < 0.001

    asyncio.run(scenario())


def stubbed_worker(monkeypatch) -> tuple[NativeRpcServer, list[bool], threading.Event]:
    """A NativeRpcServer whose daemon thread and readiness probe are faked."""
    from sharedlocalllm_backend import rpc_native
    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    ports = iter((50001, 50002))
    monkeypatch.setattr(rpc_native, "_free_port", lambda: next(ports))
    monkeypatch.setattr(
        rpc_native.socket, "create_connection", lambda *_args, **_kwargs: DummySocket()
    )

    worker = NativeRpcServer()
    runs: list[bool] = []
    release = threading.Event()

    def fake_run(include_cpu: bool, cache_dir: str | None) -> None:
        runs.append(include_cpu)
        release.wait(timeout=5)

    monkeypatch.setattr(worker, "_run", fake_run)
    return worker, runs, release


def test_rpc_worker_start_honours_the_requested_include_cpu_flag(monkeypatch) -> None:
    worker, runs, release = stubbed_worker(monkeypatch)

    # The daemon stays alive, so an unchanged request must reuse it.
    assert worker.start(False) == "127.0.0.1:50001"
    assert runs == [False]
    assert worker.start(False) == "127.0.0.1:50001"
    assert runs == [False]
    release.set()


def test_rpc_worker_restarts_when_the_cpu_opt_in_changes(monkeypatch) -> None:
    worker, runs, release = stubbed_worker(monkeypatch)

    assert worker.start(False) == "127.0.0.1:50001"
    assert worker.include_cpu is False
    assert worker.start(True) == "127.0.0.1:50002"
    assert runs == [False, True]
    assert worker.include_cpu is True
    release.set()
