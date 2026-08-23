from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any

import psutil


def _run(command: list[str], timeout: float = 3.0) -> str | None:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=flags,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu() -> dict[str, Any]:
    output = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return {"name": "No NVIDIA GPU detected", "vramTotalGb": 0.0, "vramAvailableGb": 0.0}
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {"name": first, "vramTotalGb": 0.0, "vramAvailableGb": 0.0}
    return {
        "name": parts[0],
        "vramTotalGb": round(float(parts[1]) / 1024, 2),
        "vramAvailableGb": round(float(parts[2]) / 1024, 2),
        "driverVersion": parts[3],
    }


def _adapter() -> dict[str, Any]:
    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()
    candidates: list[tuple[float, str]] = []
    for name, stat in stats.items():
        if not stat.isup or name.lower().startswith(("loopback", "lo")):
            continue
        has_ipv4 = any(addr.family == socket.AF_INET for addr in addrs.get(name, []))
        if has_ipv4:
            candidates.append((float(stat.speed or 0), name))
    if not candidates:
        return {"name": "Unknown", "kind": "other"}
    speed, name = max(candidates)
    lower = name.lower()
    kind = "wifi" if any(x in lower for x in ("wi-fi", "wifi", "wlan", "wireless")) else "ethernet"
    value: dict[str, Any] = {"name": name, "kind": kind}
    if speed > 0:
        value["linkSpeedMbps"] = speed
    return value


def probe_node(device_id: str, device_name: str, role: str = "coordinator") -> dict[str, Any]:
    memory = psutil.virtual_memory()
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine()
    return {
        "id": device_id,
        "name": device_name,
        "online": True,
        "role": role,
        "cpu": cpu,
        "ramTotalGb": round(memory.total / 1024**3, 2),
        "ramAvailableGb": round(memory.available / 1024**3, 2),
        "gpu": _gpu(),
        "adapter": _adapter(),
    }
