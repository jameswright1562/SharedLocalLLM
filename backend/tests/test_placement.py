from __future__ import annotations

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.placement import (
    distribute_layers,
    estimate_split,
    normalize_load_config,
)


def node(node_id: str, vram: float, ram: float = 32.0) -> dict:
    return {
        "id": node_id,
        "name": node_id,
        "online": True,
        "gpu": {"vramAvailableGb": vram},
        "ramAvailableGb": ram,
    }


MODEL = {
    "id": "model",
    "name": "Model",
    "sizeBytes": 4 * 1024**3,
    "contextLength": 8192,
    "layerCount": 32,
}


def test_automatic_distribution_skips_zero_vram_nodes() -> None:
    result = distribute_layers(32, [node("empty", 0), node("gpu", 8)])
    assert result == [{"nodeId": "gpu", "layers": 32, "kind": "gpu"}]


def test_known_model_with_no_available_gpu_uses_cpu_instead_of_auto_offload() -> None:
    config = normalize_load_config(
        MODEL, {"contextSize": 4096, "gpuLayers": []}, [node("local", 0)]
    )
    assert config["gpuLayers"] == []
    assert config["automaticGpuOffload"] is False


def test_context_is_clamped_and_remote_cpu_counts_toward_total() -> None:
    config = normalize_load_config(
        MODEL,
        {
            "contextSize": 99999,
            "includeRemoteCpu": True,
            "gpuLayers": [
                {"nodeId": "local", "layers": 20},
                {"nodeId": "peer", "layers": 12, "kind": "cpu"},
            ],
        },
        [node("local", 8), node("peer", 8)],
    )
    estimate = estimate_split(MODEL, config, [node("local", 8), node("peer", 8)])
    assert config["contextSize"] == 8192
    assert estimate["gpuLayers"] == 20
    assert estimate["cpuLayers"] == 0


def test_rejects_overallocated_duplicate_unknown_and_negative_splits() -> None:
    cases = [
        [
            {"nodeId": "local", "layers": 20},
            {"nodeId": "peer", "layers": 13, "kind": "cpu"},
        ],
        [{"nodeId": "local", "layers": 1}, {"nodeId": "local", "layers": 2}],
        [{"nodeId": "missing", "layers": 1}],
        [{"nodeId": "local", "layers": -1}],
    ]
    for allocations in cases:
        with pytest.raises(BackendError):
            normalize_load_config(
                MODEL,
                {
                    "contextSize": 4096,
                    "includeRemoteCpu": True,
                    "gpuLayers": allocations,
                },
                [node("local", 8), node("peer", 8)],
            )


def test_batch_size_is_clamped_to_a_safe_upper_bound() -> None:
    config = normalize_load_config(
        MODEL, {"contextSize": 4096, "batchSize": 100_000_000}, [node("local", 8)]
    )
    assert config["batchSize"] == 4096


def test_batch_size_stays_positive_and_rejects_non_numeric_input() -> None:
    zeroed = normalize_load_config(
        MODEL, {"contextSize": 4096, "batchSize": 0}, [node("local", 8)]
    )
    assert zeroed["batchSize"] == 512
    with pytest.raises(BackendError, match="Batch size"):
        normalize_load_config(
            MODEL, {"contextSize": 4096, "batchSize": "huge"}, [node("local", 8)]
        )


def test_normalize_load_config_preserves_kv_cache_options() -> None:
    config = normalize_load_config(
        MODEL,
        {
            "contextSize": 8192,
            "gpuLayers": [],
            "kvCacheK": "q8_0",
            "kvCacheV": "q4_0",
            "kvUnified": True,
        },
        [node("local", 8)],
    )
    assert config["kvCacheK"] == "q8_0"
    assert config["kvCacheV"] == "q4_0"
    assert config["kvUnified"] is True
