from __future__ import annotations

import os

from sharedlocalllm_backend.inference import build_llama_kwargs


def test_defaults_match_previous_behaviour() -> None:
    kwargs = build_llama_kwargs({}, "model.gguf", 8192, 16, None)
    cores = os.cpu_count() or 8
    assert kwargs["model_path"] == "model.gguf"
    assert kwargs["n_ctx"] == 8192
    assert kwargs["n_gpu_layers"] == 16
    assert kwargs["n_threads"] == max(1, cores // 2)
    assert kwargs["n_threads_batch"] == max(1, cores)
    assert kwargs["n_batch"] == 512
    assert kwargs["flash_attn"] is False
    assert kwargs["use_mmap"] is True
    assert kwargs["use_mlock"] is False
    assert kwargs["verbose"] is True


def test_zero_gpu_layers_offloads_automatically() -> None:
    kwargs = build_llama_kwargs({}, "model.gguf", 4096, 0, None)
    assert kwargs["n_gpu_layers"] == -1


def test_options_flow_through_to_llama() -> None:
    kwargs = build_llama_kwargs(
        {
            "flashAttention": True,
            "useMmap": False,
            "useMlock": True,
            "cpuThreads": 4,
            "batchSize": 2048,
        },
        "model.gguf", 4096, 8, [16.0, 16.0],
    )
    assert kwargs["flash_attn"] is True
    assert kwargs["use_mmap"] is False
    assert kwargs["use_mlock"] is True
    assert kwargs["n_threads"] == 4
    assert kwargs["n_batch"] == 2048
    assert kwargs["tensor_split"] == [16.0, 16.0]


def test_zero_threads_keeps_automatic_default() -> None:
    kwargs = build_llama_kwargs({"cpuThreads": 0}, "model.gguf", 4096, 0, None)
    cores = os.cpu_count() or 8
    assert kwargs["n_threads"] == max(1, cores // 2)


def test_batch_size_is_clamped_to_at_least_one() -> None:
    kwargs = build_llama_kwargs({"batchSize": 0}, "model.gguf", 4096, 0, None)
    assert kwargs["n_batch"] == 1
