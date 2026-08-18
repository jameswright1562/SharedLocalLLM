from __future__ import annotations

import ctypes

import pytest

from sharedlocalllm_backend.rpc_native import (
    classify_devices,
    prepare_model_devices,
    tensor_split_for,
)


def device(name: str, type_: int = 1, registry: str = "CUDA", pointer: int = 0) -> dict:
    return {"name": name, "type": type_, "registry": registry, "pointer": pointer}


@pytest.fixture(autouse=True)
def clear_pending_devices():
    from sharedlocalllm_backend import rpc_native

    rpc_native._pending_devices[:] = []
    yield
    rpc_native._pending_devices[:] = []


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
    assert tensor_split_for(1, 1, 16, 16) == [16.0, 16.0]


def test_tensor_split_spreads_across_multiple_devices() -> None:
    assert tensor_split_for(2, 1, 12, 6) == [6.0, 6.0, 6.0]
    assert tensor_split_for(0, 2, 0, 10) == [5.0, 5.0]
    assert tensor_split_for(1, 0, 8, 0) == [8.0]


def test_pending_devices_are_injected_into_model_params() -> None:
    from sharedlocalllm_backend.rpc_native import _install_model_devices_patch

    import llama_cpp
    import llama_cpp._internals as internals

    _install_model_devices_patch()
    prepare_model_devices([123, 456])
    params = llama_cpp.llama_model_default_params()
    with pytest.raises(ValueError):
        internals.LlamaModel(path_model="does-not-exist.gguf", params=params, verbose=False)
    array = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
    assert array[0] == 123
    assert array[1] == 456
    assert array[2] is None


def test_prepare_rpc_load_distributed_builds_remote_first_split() -> None:
    import asyncio

    from sharedlocalllm_backend.rpc_native import NativeRpcServer, prepare_rpc_load

    async def scenario() -> None:
        server = NativeRpcServer()
        endpoint = await asyncio.to_thread(server.start)
        split = await asyncio.to_thread(prepare_rpc_load, endpoint, 16, 16)
        assert split is not None
        assert split[0] == 16.0
        assert abs(sum(split) - 32.0) < 0.001

    asyncio.run(scenario())
