from __future__ import annotations

import json
import os
import secrets
import threading
import uuid
from pathlib import Path
from typing import Any


class Store:
    def __init__(self) -> None:
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        self.data_dir = Path(local) / "SharedLocalLLM"
        self.logs_dir = self.data_dir / "logs"
        self.path = self.data_dir / "python-backend.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._settings = self._load()
        self._logs: list[str] = ["READY Python backend initialized"]

    def _defaults(self) -> dict[str, Any]:
        return {
            "installId": f"device-{uuid.uuid4()}",
            "deviceName": os.environ.get("COMPUTERNAME") or "SharedLocalLLM PC",
            "setupComplete": False,
            "apiPort": 11435,
            "apiKey": secrets.token_urlsafe(32),
            "authRequired": True,
            "autostart": False,
            "customModelDirectories": [],
            "modelLoadConfigs": {},
            "modelTunes": {},
            # Canonical multi-peer list (up to N computers, e.g. 2 Windows + 1
            # Mac). "peer" is a legacy compat mirror of peers[0], kept so older
            # readers/tests keep working; new code must use "peers".
            "peers": [],
            "peer": None,
            "benchmarks": [],
        }

    def _load(self) -> dict[str, Any]:
        defaults = self._defaults()
        loaded = self._read_json(self.path)
        if not loaded:
            loaded = self._legacy_settings()
        if loaded:
            defaults.update(loaded)
        defaults["peers"] = _normalize_peers(defaults.get("peers"), defaults.get("peer"))
        defaults["peer"] = defaults["peers"][0] if defaults["peers"] else None
        self._write(defaults)
        return defaults

    def _legacy_settings(self) -> dict[str, Any] | None:
        legacy = self._read_json(self.data_dir / "settings.json")
        if not legacy:
            return None
        migrated: dict[str, Any] = {}
        for key in (
            "installId", "deviceName", "setupComplete", "apiPort", "autostart",
            "customModelDirectories", "benchmarks",
        ):
            if key in legacy and legacy[key] is not None:
                migrated[key] = legacy[key]
        peers = legacy.get("peers")
        if isinstance(peers, list) and peers:
            migrated["peers"] = [peer for peer in peers if isinstance(peer, dict)]
        elif isinstance(legacy.get("peer"), dict):
            migrated["peers"] = [legacy["peer"]]
        return migrated

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, value: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._settings.get(key, default)

    def update(self, **values: Any) -> None:
        with self._lock:
            self._settings.update(values)
            self._write(self._settings)

    def append_benchmark(self, value: dict[str, Any]) -> None:
        with self._lock:
            values = list(self._settings.get("benchmarks", []))
            values.append(value)
            self._settings["benchmarks"] = values[-100:]
            self._write(self._settings)

    def save_model_load_config(self, model_id: str, config: dict[str, Any]) -> None:
        """Remember the exact load configuration that launched a model.

        The next launch of the same model restores these values instead of
        falling back to defaults.
        """
        with self._lock:
            configs = dict(self._settings.get("modelLoadConfigs") or {})
            configs[model_id] = config
            self._settings["modelLoadConfigs"] = configs
            self._write(self._settings)

    def model_load_configs(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._settings.get("modelLoadConfigs") or {})

    def save_model_tune(self, model_id: str, tune: dict[str, Any]) -> None:
        """Remember the latest autotune result for a model.

        Results carry the topology fingerprint they were measured on; callers
        must re-check it before applying the winners to a launch.
        """
        with self._lock:
            tunes = dict(self._settings.get("modelTunes") or {})
            tunes[model_id] = tune
            self._settings["modelTunes"] = tunes
            self._write(self._settings)

    def model_tunes(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._settings.get("modelTunes") or {})

    def log(self, level: str, event: str, detail: str) -> None:
        line = f"{level} {event}: {detail}"
        with self._lock:
            self._logs.append(line)
            del self._logs[:-250]
        try:
            with (self.logs_dir / "python-backend.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def logs(self) -> list[str]:
        with self._lock:
            return list(self._logs)


def save_peers(store: "Store", peers: list[dict[str, Any]]) -> None:
    """Persist the multi-peer list, keeping the legacy "peer" mirror in sync.

    "peers" is the source of truth (up to N trusted computers); "peer" mirrors
    peers[0] so older readers and tests keep working.
    """
    cleaned = [peer for peer in peers if isinstance(peer, dict) and peer.get("id")]
    store.update(peers=cleaned, peer=cleaned[0] if cleaned else None)


def stored_peers(store: Any) -> list[dict[str, Any]]:
    """Every paired computer from any store-like object (real or test fake).

    Reads the canonical "peers" list with a fallback to the legacy single
    "peer" record, so older persisted settings and test fixtures keep working.
    """
    try:
        stored = store.get("peers", [])
    except Exception:
        stored = []
    if isinstance(stored, list) and stored:
        return [peer for peer in stored if isinstance(peer, dict) and peer.get("id")]
    try:
        legacy = store.get("peer")
    except Exception:
        legacy = None
    # Legacy records (and minimal test doubles) may carry only an address.
    if isinstance(legacy, dict) and (legacy.get("id") or legacy.get("address")):
        return [legacy]
    return []


def _normalize_peers(peers: Any, legacy: Any) -> list[dict[str, Any]]:
    if isinstance(peers, list) and peers:
        return [peer for peer in peers if isinstance(peer, dict) and peer.get("id")]
    if isinstance(legacy, dict) and legacy.get("id"):
        return [legacy]
    return []
