from __future__ import annotations

import asyncio

from sharedlocalllm_backend.runtime import BackendRuntime


class SettingsStore:
    def __init__(self) -> None:
        self.values: dict = {"apiPort": 11435, "apiKey": "k", "authRequired": True}

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def update(self, **values) -> None:
        self.values.update(values)

    def model_load_configs(self) -> dict:
        return {}

    def logs(self) -> list[str]:
        return []


def settings_runtime() -> BackendRuntime:
    runtime = BackendRuntime.__new__(BackendRuntime)
    runtime.store = SettingsStore()
    runtime.local_node = {"id": "local", "name": "Old name"}
    runtime.models = []
    runtime.modelDirectories = []
    runtime.network = None
    runtime.cluster = {"status": "idle"}
    runtime._runtime = {"status": "ready"}
    runtime.api_health = None
    runtime.api_port_changed = None
    return runtime


def test_update_settings_persists_the_authentication_toggle() -> None:
    runtime = settings_runtime()

    snapshot = asyncio.run(
        runtime.update_settings(
            {"deviceName": "PC-1", "apiPort": 12000, "autostart": False, "authRequired": False}
        )
    )

    assert runtime.store.values["authRequired"] is False
    assert snapshot["authRequired"] is False
    assert runtime.get_api_config()["authRequired"] is False


def test_update_settings_defaults_to_requiring_the_bearer_key() -> None:
    runtime = settings_runtime()
    runtime.store.values["authRequired"] = False

    snapshot = asyncio.run(
        runtime.update_settings({"deviceName": "PC-1", "apiPort": 12000})
    )

    assert runtime.store.values["authRequired"] is True
    assert snapshot["authRequired"] is True
