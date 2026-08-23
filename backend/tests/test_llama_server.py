from __future__ import annotations

import json
import logging
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.llama_server import (
    binary_version,
    find_manifest,
    load_manifest,
    locate_llama_server,
    probe_health,
)


def repo_manifest_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "public" / "runtime" / "llama-cpp-manifest.json"
    if not path.is_file():
        pytest.skip("repository runtime manifest is not present")
    return path


def test_repository_manifest_loads_and_pins_the_llama_server() -> None:
    manifest = load_manifest(repo_manifest_path())
    assert manifest["channel"] == "pinned"
    assert "llama-server.exe" in manifest["requiredExecutables"]
    assert manifest["llamaServer"]["entry"] == "llama-server.exe"
    assert manifest["llamaServer"]["assetKey"] in manifest["release"]


def test_find_manifest_discovers_the_repository_copy() -> None:
    assert find_manifest() == repo_manifest_path()


def write_manifest(tmp_path: Path, **overrides) -> Path:
    manifest: dict = {
        "schemaVersion": 1,
        "enabled": True,
        "channel": "pinned",
        "release": {
            "llamaCpp": {
                "url": (
                    "https://github.com/ggml-org/llama.cpp/releases/download/"
                    "b10405/llama-b10405-bin-win-cuda-12.4-x64.zip"
                ),
                "size": 123,
                "sha256": "a" * 64,
            }
        },
        "llamaServer": {"assetKey": "llamaCpp", "entry": "llama-server.exe"},
    }
    manifest.update(overrides)
    path = tmp_path / "llama-cpp-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_manifest_rejects_disabled_channel_and_tampered_assets(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="disabled"):
        load_manifest(write_manifest(tmp_path / "a", enabled=False))

    wrong_channel = write_manifest(tmp_path / "b")
    payload = json.loads(wrong_channel.read_text(encoding="utf-8"))
    payload["channel"] = "latest"
    wrong_channel.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BackendError, match="pinned"):
        load_manifest(wrong_channel)

    with pytest.raises(BackendError, match="origin"):
        load_manifest(
            write_manifest(
                tmp_path / "c",
                release={
                    "llamaCpp": {
                        "url": "http://example.invalid/llama.zip",
                        "size": 1,
                        "sha256": "a" * 64,
                    }
                },
            )
        )

    with pytest.raises(BackendError, match="SHA-256"):
        bad_digest = write_manifest(tmp_path / "d")
        payload = json.loads(bad_digest.read_text(encoding="utf-8"))
        payload["release"]["llamaCpp"]["sha256"] = "A" * 64
        bad_digest.write_text(json.dumps(payload), encoding="utf-8")
        load_manifest(bad_digest)

    with pytest.raises(BackendError, match="size"):
        load_manifest(
            write_manifest(
                tmp_path / "e",
                release={
                    "llamaCpp": {
                        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b1/x.zip",
                        "size": 0,
                        "sha256": "a" * 64,
                    }
                },
            )
        )


def test_locate_llama_server_returns_the_first_directory_that_has_it(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    filled = tmp_path / "filled"
    empty.mkdir()
    filled.mkdir()
    (filled / "llama-server.exe").write_bytes(b"MZ")

    assert locate_llama_server((empty, filled)) == filled / "llama-server.exe"
    assert locate_llama_server((empty,)) is None


def test_binary_version_runs_version_flag_and_parses_first_line() -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="version: 10405 (abc123)\nbuilt with MSVC\n")

    version = binary_version(Path("anywhere/llama-server.exe"), runner=fake_runner)

    assert calls == [["anywhere\\llama-server.exe", "--version"]] or calls == [
        ["anywhere/llama-server.exe", "--version"]
    ]
    assert version == "version: 10405 (abc123)"


def test_binary_version_treats_failures_as_unhealthy() -> None:
    def failing_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError("not an executable")

    assert binary_version(Path("missing.exe"), runner=failing_runner) is None


class _HealthHandler(BaseHTTPRequestHandler):
    status_code = 200
    body = b'{"status":"ok"}'
    require_key = False

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.require_key and self.headers.get("Authorization") != "Bearer secret":
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *_args) -> None:
        logging.getLogger(__name__).debug("health probe: %s", self.requestline)


@pytest.fixture()
def health_server():
    factory = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=factory.serve_forever, daemon=True)
    thread.start()
    try:
        yield factory
    finally:
        factory.shutdown()
        factory.server_close()


def test_probe_health_accepts_a_healthy_loopback_instance(health_server: HTTPServer) -> None:
    port = health_server.server_address[1]
    assert probe_health(port, api_key="secret") is True


def test_probe_health_rejects_wrong_status_or_missing_key() -> None:
    rejecting = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    rejecting.RequestHandlerClass.require_key = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=rejecting.serve_forever, daemon=True)
    thread.start()
    try:
        port = rejecting.server_address[1]
        assert probe_health(port) is False
        assert probe_health(port, api_key="secret") is True
    finally:
        rejecting.shutdown()
        rejecting.server_close()

    unhealthy = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    unhealthy.RequestHandlerClass.status_code = 503  # type: ignore[attr-defined]
    thread = threading.Thread(target=unhealthy.serve_forever, daemon=True)
    thread.start()
    try:
        assert probe_health(unhealthy.server_address[1]) is False
    finally:
        unhealthy.shutdown()
        unhealthy.server_close()
