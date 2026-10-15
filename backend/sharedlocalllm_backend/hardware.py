from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
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


def _gpu(total_ram_gb: float, available_ram_gb: float) -> dict[str, Any]:
    output = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if output:
        first = output.splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 4:
            return {
                "name": parts[0],
                "vramTotalGb": round(float(parts[1]) / 1024, 2),
                "vramAvailableGb": round(float(parts[2]) / 1024, 2),
                "driverVersion": parts[3],
            }
        return {"name": first, "vramTotalGb": 0.0, "vramAvailableGb": 0.0}
    apple = _apple_silicon_gpu(total_ram_gb, available_ram_gb)
    if apple is not None:
        return apple
    return {"name": "No GPU detected", "vramTotalGb": 0.0, "vramAvailableGb": 0.0}


def _apple_silicon_gpu(total_ram_gb: float, available_ram_gb: float) -> dict[str, Any] | None:
    """Apple Silicon (including M5) unified-memory GPU for Metal inference.

    No nvidia-smi exists on macOS; the GPU shares the system RAM pool, so a
    conservative 75% of currently available RAM is reported as usable VRAM
    instead of "no GPU". Intel Macs keep the generic fallback.
    """
    if sys.platform != "darwin":
        return None
    arch = (_run(["uname", "-m"]) or "").strip()
    if arch != "arm64" or total_ram_gb <= 0:
        return None
    chip = (_run(["sysctl", "-n", "hw.model"]) or "").strip() or "Apple Silicon"
    os_version = (_run(["sw_vers", "-productVersion"]) or "").strip()
    available = round(min(available_ram_gb * 0.75, total_ram_gb), 2)
    info: dict[str, Any] = {
        "name": f"{chip} GPU (unified memory, Metal)",
        "vramTotalGb": round(total_ram_gb, 2),
        "vramAvailableGb": available,
    }
    if os_version:
        info["driverVersion"] = f"macOS {os_version}"
    return info


def _adapter() -> dict[str, Any]:
    if sys.platform == "darwin":
        mac = _macos_adapter()
        if mac is not None:
            return mac
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


def _macos_adapter() -> dict[str, Any] | None:
    """Map the default-route interface (e.g. en0) to its hardware port."""
    route = _run(["route", "-n", "get", "default"])
    interface = None
    for line in (route or "").splitlines():
        text = line.strip()
        if text.startswith("interface:"):
            interface = text.split(":", 1)[1].strip()
    if not interface:
        return None
    ports = _run(["networksetup", "-listallhardwareports"], timeout=4.0)
    if ports:
        current_port = ""
        for line in ports.splitlines():
            text = line.strip()
            if text.startswith("Hardware Port:"):
                current_port = text.split(":", 1)[1].strip()
            elif text.startswith("Device:") and text.split(":", 1)[1].strip() == interface:
                lowered = current_port.lower()
                if "wi-fi" in lowered or "airport" in lowered:
                    kind = "wifi"
                elif "ethernet" in lowered or "thunderbolt" in lowered:
                    kind = "ethernet"
                else:
                    kind = "other"
                return {"name": f"{current_port} ({interface})", "kind": kind}
    return {"name": interface, "kind": "other"}


def _device_name() -> str:
    for candidate in (
        os.environ.get("COMPUTERNAME"), os.environ.get("HOSTNAME"),
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    for command in (["scutil", "--get", "ComputerName"], ["hostname"]):
        name = _run(command)
        if name:
            return name
    return "SharedLocalLLM PC"


def probe_node(device_id: str, device_name: str, role: str = "coordinator") -> dict[str, Any]:
    memory = psutil.virtual_memory()
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine()
    total_gb = round(memory.total / 1024**3, 2)
    available_gb = round(memory.available / 1024**3, 2)
    return {
        "id": device_id,
        "name": device_name or _device_name(),
        "online": True,
        "role": role,
        "cpu": cpu,
        "ramTotalGb": total_gb,
        "ramAvailableGb": available_gb,
        "gpu": _gpu(total_gb, available_gb),
        "adapter": _adapter(),
    }
