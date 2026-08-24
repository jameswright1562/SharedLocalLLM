from __future__ import annotations

import asyncio
from typing import cast

from sharedlocalllm_backend.runtime import BackendRuntime
from sharedlocalllm_backend.store import Store


class SettingsStore:
    def __init__(self) -> None:
        self.values: dict = {"apiPort": 11435, "apiKey": "k", "authRequired": True}

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def update(self, **values) -> None:
        self.values.update(values)

    def model_load_configs(self) -> dict:
        return {}

    def model_tunes(self) -> dict:
        return {}

    def logs(self) -> list[str]:
        return []


def settings_runtime() -> tuple[BackendRuntime, SettingsStore]:
    runtime = BackendRuntime.__new__(BackendRuntime)
    store = SettingsStore()
    runtime.store = cast(Store, store)
    runtime.local_node = {"id": "local", "name": "Old name"}
    runtime.models = []
    runtime.network = None
    runtime.cluster = {"status": "idle"}
    runtime._runtime = {"status": "ready"}
    runtime.api_health = None
    runtime.api_port_changed = None
    return runtime, store


def test_update_settings_persists_the_authentication_toggle() -> None:
    runtime, store = settings_runtime()

    snapshot = asyncio.run(
        runtime.update_settings(
            {"deviceName": "PC-1", "apiPort": 12000, "autostart": False, "authRequired": False}
        )
    )

    assert store.values["authRequired"] is False
    assert snapshot["authRequired"] is False
    assert runtime.get_api_config()["authRequired"] is False


def test_update_settings_defaults_to_requiring_the_bearer_key() -> None:
    runtime, store = settings_runtime()
    store.values["authRequired"] = False

    snapshot = asyncio.run(
        runtime.update_settings({"deviceName": "PC-1", "apiPort": 12000})
    )

    assert store.values["authRequired"] is True
    assert snapshot["authRequired"] is True
