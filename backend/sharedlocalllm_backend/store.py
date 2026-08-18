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
            "installId": str(uuid.uuid4()),
            "deviceName": os.environ.get("COMPUTERNAME") or "SharedLocalLLM PC",
            "setupComplete": False,
            "apiPort": 11435,
            "apiKey": secrets.token_urlsafe(32),
            "autostart": False,
            "customModelDirectories": [],
            "peer": None,
            "benchmarks": [],
        }

    def _load(self) -> dict[str, Any]:
        defaults = self._defaults()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
        self._write(defaults)
        return defaults

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
