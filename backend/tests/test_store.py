from __future__ import annotations

import json

from sharedlocalllm_backend.store import Store


def test_migrates_legacy_rust_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    data = tmp_path / "SharedLocalLLM"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({
        "installId": "device-existing",
        "deviceName": "Node A",
        "setupComplete": True,
        "apiPort": 12000,
        "customModelDirectories": ["C:/Models"],
        "peers": [{"id": "peer-a", "name": "Node B", "address": "10.10.10.2:49158"}],
    }), encoding="utf-8")
    store = Store()
    assert store.get("installId") == "device-existing"
    assert store.get("deviceName") == "Node A"
    assert store.get("apiPort") == 12000
    assert store.get("peer")["id"] == "peer-a"
    assert (data / "python-backend.json").is_file()


def test_model_tunes_persist_per_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    store = Store()

    store.save_model_tune("model-a", {"winners": {"batchSize": 2048}, "fingerprint": "abc"})
    reloaded = Store()

    assert reloaded.model_tunes()["model-a"]["fingerprint"] == "abc"
    assert "model-b" not in reloaded.model_tunes()
