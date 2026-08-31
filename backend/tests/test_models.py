from __future__ import annotations

import struct
from pathlib import Path

from sharedlocalllm_backend.gguf import (
    chat_template_supports_reasoning_preserve,
    has_nextn_tensors,
)
from sharedlocalllm_backend.models import _fit, discover_local, hf_cache_roots, merge_remote


def node(node_id: str, vram: float = 12.0) -> dict:
    return {
        "id": node_id,
        "name": node_id,
        "online": True,
        "gpu": {"vramAvailableGb": vram},
        "ramAvailableGb": 32.0,
    }


def test_fit_sizes_kv_from_default_launch_context_not_native_maximum() -> None:
    hardware = node("node-a", vram=24.0)
    metadata = {"layerCount": 64, "contextLength": 262144}
    size = 18 * 1024**3

    assert _fit(size, hardware, None, metadata) == "single-node"


def test_fit_still_reports_does_not_fit_when_weights_alone_exceed_vram() -> None:
    hardware = node("node-a", vram=24.0)
    metadata = {"layerCount": 64, "contextLength": 262144}

    assert _fit(31 * 1024**3, hardware, None, metadata) in ("combined-gpu", "gpu-ram")
    assert _fit(80 * 1024**3, hardware, None, metadata) == "does-not-fit"


def test_discovers_a_gguf_and_extracts_quantization(tmp_path: Path) -> None:
    path = tmp_path / "Qwen-Test-Q6_K.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 4096)
    models, paths = discover_local([tmp_path], "node-a", node("node-a"))
    assert len(models) == 1
    assert models[0]["quantization"] == "Q6_K"
    assert paths[models[0]["id"]] == str(path)


def test_merge_marks_remote_only_models() -> None:
    local = [{"id": "a", "name": "A", "locations": [], "remoteOnly": False}]
    remote = [{"id": "b", "name": "B", "locations": []}]
    merged = merge_remote(local, remote)
    assert {model["id"] for model in merged} == {"a", "b"}
    assert next(model for model in merged if model["id"] == "b")["remoteOnly"] is True


def test_hf_cache_roots_returns_snapshots_containing_gguf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snapshot = tmp_path / ".cache" / "huggingface" / "hub" / "models--a--b" / "snapshots" / "abc123"
    (snapshot / "Q6_K").mkdir(parents=True)
    (snapshot / "Q6_K" / "model-00001-of-00002.gguf").write_bytes(b"GGUF")
    (snapshot / "Q6_K" / "model-00002-of-00002.gguf").write_bytes(b"GGUF")
    (tmp_path / ".cache" / "huggingface" / "hub" / "models--c--d" / "snapshots" / "def456").mkdir(parents=True)

    roots = hf_cache_roots()
    assert roots == [snapshot]


def test_hf_cache_roots_ignores_cache_without_gguf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".cache" / "huggingface" / "hub" / "models--x--y" / "snapshots" / "aaa").mkdir(parents=True)
    assert hf_cache_roots() == []



def test_discovery_marks_locally_stored_models_as_local(tmp_path: Path) -> None:
    (tmp_path / "Qwen-Test-Q6_K.gguf").write_bytes(b"GGUF" + b"\0" * 4096)
    models, _ = discover_local([tmp_path], "node-a", node("node-a"))
    assert models[0]["isLocal"] is True


def test_merge_reports_which_models_live_on_this_computer() -> None:
    local = [
        {
            "id": "a",
            "name": "A",
            "locations": [{"nodeId": "node-a", "path": "a.gguf", "source": "custom"}],
            "remoteOnly": False,
        }
    ]
    remote = [
        {
            "id": "b",
            "name": "B",
            "locations": [{"nodeId": "node-b", "path": "b.gguf", "source": "custom"}],
        },
        {
            "id": "a",
            "name": "A",
            "locations": [{"nodeId": "node-b", "path": "a-copy.gguf", "source": "custom"}],
        },
    ]
    merged = merge_remote(local, remote, local_node_id="node-a")
    by_id = {model["id"]: model for model in merged}
    assert by_id["a"]["isLocal"] is True
    assert [loc["nodeId"] for loc in by_id["a"]["locations"]] == ["node-a", "node-b"]
    assert by_id["b"]["isLocal"] is False
    assert by_id["b"]["remoteOnly"] is True


def test_discovery_rejects_invalid_incomplete_and_projector_files(tmp_path: Path) -> None:
    (tmp_path / "broken.gguf").write_bytes(b"not a GGUF")
    (tmp_path / "model-00001-of-00003.gguf").write_bytes(b"GGUF")
    (tmp_path / "mmproj-model.gguf").write_bytes(b"GGUF")
    models, _ = discover_local([tmp_path], "node-a", node("node-a"))
    assert models == []


def test_discovery_accepts_a_complete_shard_set(tmp_path: Path) -> None:
    for index in (1, 2):
        (tmp_path / f"model-0000{index}-of-00002.gguf").write_bytes(b"GGUF")
    models, _ = discover_local([tmp_path], "node-a", node("node-a"))
    assert len(models) == 1
    assert models[0]["shards"] == 2


def test_detects_embedded_nextn_tensors_without_reading_tensor_data(tmp_path: Path) -> None:
    name = b"blk.0.nextn_predictor.weight"
    gguf = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 0)
        + struct.pack("<Q", len(name)) + name
        + struct.pack("<I", 1)
        + struct.pack("<QIQ", 16, 0, 0)
    )
    path = tmp_path / "qwen-mtp.gguf"
    path.write_bytes(gguf)

    assert has_nextn_tensors(path) is True
    models, _ = discover_local([tmp_path], "node-a", node("node-a"))
    assert models[0]["mtp"] is True


def test_detects_reasoning_preservation_from_the_gguf_chat_template(tmp_path: Path) -> None:
    key = b"tokenizer.chat_template"
    template = (
        b"{% if preserve_reasoning %}"
        b"{{ message.reasoning_content }}"
        b"{% endif %}"
    )
    gguf = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", len(key)) + key
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(template)) + template
    )
    path = tmp_path / "reasoning.gguf"
    path.write_bytes(gguf)

    models, _ = discover_local([tmp_path], "node-a", node("node-a"))

    assert models[0]["reasoningPreserve"] is True
    assert chat_template_supports_reasoning_preserve("{{ message.content }}") is False
