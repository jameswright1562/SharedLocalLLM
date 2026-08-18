from __future__ import annotations

from pathlib import Path

from sharedlocalllm_backend.models import discover_local, merge_remote


def node(node_id: str, vram: float = 12.0) -> dict:
    return {
        "id": node_id,
        "name": node_id,
        "online": True,
        "gpu": {"vramAvailableGb": vram},
        "ramAvailableGb": 32.0,
    }


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
