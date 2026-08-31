"""Pinned llama-server runtime plumbing (dual-engine plan, phase 1).

Read-only today: locate the pinned binary, validate the runtime manifest, and
health-check a running instance. Launching llama-server arrives in phase 2.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .errors import BackendError

MANIFEST_FILENAME = "llama-cpp-manifest.json"
SERVER_EXECUTABLE = "llama-server.exe"
ALLOWED_ASSET_PREFIX = "https://github.com/ggml-org/llama.cpp/releases/download/"
EXPERIMENTAL_DIR_ENV = "SHAREDLOCALLLM_LLAMA_SERVER_DIR"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


def manifest_candidates() -> list[Path]:
    """Manifest locations in repo checkouts and beside an installed sidecar."""
    repo_copy = Path(__file__).resolve().parents[2] / "public" / "runtime"
    executable_dir = Path(sys.executable).resolve().parent
    return [
        repo_copy,
        executable_dir / "runtime",
        executable_dir,
    ]


def find_manifest() -> Path | None:
    for directory in manifest_candidates():
        candidate = directory / MANIFEST_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _validate_asset(key: str, asset: Any) -> None:
    if not isinstance(asset, dict):
        raise BackendError("runtime_manifest_invalid", f"Release asset '{key}' is malformed.")
    url = str(asset.get("url", ""))
    if not url.startswith(ALLOWED_ASSET_PREFIX):
        raise BackendError(
            "runtime_manifest_insecure_origin",
            f"Asset '{key}' origin is not an official llama.cpp release "
            f"({ALLOWED_ASSET_PREFIX}).",
        )
    digest = str(asset.get("sha256", ""))
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise BackendError(
            "runtime_manifest_bad_digest",
            f"Asset '{key}' must pin a 64-character lowercase SHA-256 digest.",
        )
    size = asset.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise BackendError(
            "runtime_manifest_bad_size",
            f"Asset '{key}' must pin a positive byte size.",
        )


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate the pinned runtime manifest."""
    if not path.is_file():
        raise BackendError(
            "runtime_manifest_missing", f"The runtime manifest is missing at {path}."
        )
    try:
        raw = path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
    except OSError as error:
        raise BackendError("runtime_manifest_unreadable", str(error)) from error
    except json.JSONDecodeError as error:
        raise BackendError("runtime_manifest_invalid", f"The runtime manifest is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise BackendError("runtime_manifest_invalid", "The runtime manifest must be a JSON object.")
    if manifest.get("enabled") is not True:
        raise BackendError("runtime_manifest_disabled", "The runtime manifest is disabled.")
    if manifest.get("channel") != "pinned":
        raise BackendError(
            "runtime_manifest_channel",
            "Only the 'pinned' channel is accepted; never float to latest releases.",
        )
    release = manifest.get("release")
    if not isinstance(release, dict) or not release:
        raise BackendError("runtime_manifest_invalid", "The runtime manifest has no pinned release.")
    asset_keys = [
        key
        for key, value in release.items()
        if isinstance(value, dict)
    ]
    if not asset_keys:
        raise BackendError("runtime_manifest_invalid", "The runtime manifest pins no download assets.")
    for key in asset_keys:
        _validate_asset(key, release[key])
    server = manifest.get("llamaServer")
    if server is not None:
        if not isinstance(server, dict) or server.get("entry") != SERVER_EXECUTABLE:
            raise BackendError(
                "runtime_manifest_llama_server",
                f"The llamaServer block must name the '{SERVER_EXECUTABLE}' entry.",
            )
        asset_key = server.get("assetKey")
        if asset_key and asset_key not in release:
            raise BackendError(
                "runtime_manifest_llama_server",
                f"The llamaServer block points at unknown asset '{asset_key}'.",
            )
    return manifest


def locate_llama_server(search_dirs: list[Path] | tuple[Path, ...]) -> Path | None:
    """First directory that actually contains the pinned server executable."""
    for directory in search_dirs:
        candidate = Path(directory) / SERVER_EXECUTABLE
        if candidate.is_file():
            return candidate
    return None


def binary_version(
    exe: Path,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> str | None:
    """Executable health check: run ``--version`` and return its first line.

    The runner is injectable so tests never execute a real binary.
    """
    run = runner or (
        lambda command: subprocess.run(command, capture_output=True, text=True, timeout=20)
    )
    try:
        result = run([str(exe), "--version"])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    output = str(getattr(result, "stdout", "") or "")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("version"):
            return line
    return lines[0] if lines else None


def probe_health(
    port: int,
    api_key: str | None = None,
    host: str = "127.0.0.1",
    timeout: float = 3.0,
) -> bool:
    """True when a loopback llama-server answers its health endpoint correctly."""
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        connection.request("GET", "/health", headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            return False
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
        return isinstance(payload, dict) and payload.get("status") == "ok"
    except (OSError, ValueError):
        return False
    finally:
        connection.close()


def install_root_candidates() -> list[Path]:
    """Where the verified installer drops the binary, checked beside the app.

    By default only directories populated by the SHA-256-verified installer are
    searched. An explicit, opt-in override (``SHAREDLOCALLLM_LLAMA_SERVER_DIR``)
    prepends a user-supplied directory so an experimental build (for example the
    Unsloth ``glm5next`` fork that mainline llama.cpp has not yet merged) can be
    exercised without touching the pinned manifest. Setting this variable is an
    explicit escape hatch: the binary there is NOT digest-verified against the
    official release, so it must only be used for throwaway experiments.
    """
    override = os.environ.get(EXPERIMENTAL_DIR_ENV)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    executable_dir = Path(sys.executable).resolve().parent
    repo_dir = Path(__file__).resolve().parents[2]
    candidates += [
        repo_dir / "backend" / "runtime" / "llama-bin",
        executable_dir / "runtime" / "llama-bin",
        executable_dir / "llama-bin",
    ]
    return candidates
