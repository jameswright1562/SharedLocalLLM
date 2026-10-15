from __future__ import annotations

import json

from sharedlocalllm_backend.store import Store, save_peers, stored_peers


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
    assert [peer["id"] for peer in store.get("peers")] == ["peer-a"]
    assert (data / "python-backend.json").is_file()


def test_legacy_migration_keeps_every_paired_computer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    data = tmp_path / "SharedLocalLLM"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({
        "installId": "device-existing",
        "peers": [
            {"id": "peer-a", "name": "Node B", "address": "10.10.10.2:49158"},
            {"id": "peer-b", "name": "Node C", "address": "10.10.10.3:49158"},
        ],
    }), encoding="utf-8")
    store = Store()
    assert [peer["id"] for peer in store.get("peers")] == ["peer-a", "peer-b"]
    assert store.get("peer")["id"] == "peer-a"


def test_save_peers_updates_the_list_and_the_legacy_mirror(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    store = Store()
    save_peers(store, [
        {"id": "peer-a", "name": "Node B", "address": "10.10.10.2:49158"},
        {"id": "peer-b", "name": "Node C", "address": "10.10.10.3:49158"},
    ])
    assert [peer["id"] for peer in stored_peers(store)] == ["peer-a", "peer-b"]
    assert store.get("peer")["id"] == "peer-a"
    reloaded = Store()
    assert [peer["id"] for peer in stored_peers(reloaded)] == ["peer-a", "peer-b"]


def test_model_tunes_persist_per_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    store = Store()

    store.save_model_tune("model-a", {"winners": {"batchSize": 2048}, "fingerprint": "abc"})
    reloaded = Store()

    assert reloaded.model_tunes()["model-a"]["fingerprint"] == "abc"
    assert "model-b" not in reloaded.model_tunes()
