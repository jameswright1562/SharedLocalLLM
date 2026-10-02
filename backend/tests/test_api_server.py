from __future__ import annotations

import asyncio
import errno
import os
import socket
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from sharedlocalllm_backend.api_server import ApiServerManager, start_with_port_fallback
from sharedlocalllm_backend.errors import BackendError


def test_real_api_server_recovers_from_an_occupied_loopback_port() -> None:
    # Uvicorn raises SystemExit in its server task on a failed bind. Keep this
    # regression in a child process so a failure cannot kill the pytest loop.
    script = dedent('''
        import asyncio
        import socket
        from types import SimpleNamespace
        import httpx
        from sharedlocalllm_backend.api_server import ApiServerManager, start_with_port_fallback

        class Store:
            def __init__(self):
                self.values = {"apiKey": "test-key", "authRequired": True}
            def get(self, key, default=None):
                return self.values.get(key, default)
            def update(self, **values):
                self.values.update(values)
            def log(self, *_args):
                pass

        async def scenario(preferred):
            store = Store()
            manager = ApiServerManager(SimpleNamespace(store=store))
            try:
                port = await start_with_port_fallback(manager, store, preferred)
                assert port != preferred
                assert store.get("apiPort") == port
                assert manager.is_healthy()
                async with httpx.AsyncClient() as client:
                    url = f"http://127.0.0.1:{port}/health"
                    assert (await client.get(url)).status_code == 401
                    response = await client.get(url, headers={"Authorization": "Bearer test-key"})
                    assert response.json() == {"status": "ok"}
            finally:
                await manager.stop()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            asyncio.run(scenario(occupied.getsockname()[1]))
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("error", [
    BackendError("api_start_timeout", "diagnostic startup timeout"),
    OSError(errno.EIO, "diagnostic I/O failure"),
])
def test_fallback_preserves_non_binding_startup_failures(error: Exception) -> None:
    class FailingStarter:
        calls = 0

        async def start(self, port: int) -> None:
            self.calls += 1
            raise error

    starter = FailingStarter()
    with pytest.raises(type(error)) as caught:
        asyncio.run(start_with_port_fallback(starter, {}, 11435))
    assert caught.value is error
    assert starter.calls == 1


def test_failed_server_task_releases_the_reserved_socket(monkeypatch) -> None:
    ports: list[int] = []

    class FailingServer:
        started = False
        should_exit = False

        def __init__(self, _config) -> None:
            pass

        async def serve(self, sockets: list[socket.socket]) -> None:
            ports.append(sockets[0].getsockname()[1])
            raise RuntimeError("diagnostic server failure")

    monkeypatch.setattr("sharedlocalllm_backend.api_server.uvicorn.Server", FailingServer)
    manager = ApiServerManager(SimpleNamespace())
    with pytest.raises(RuntimeError, match="diagnostic"):
        asyncio.run(manager.start(0))
    assert manager.port is None
    assert manager.task is None
    assert manager.server is None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as released:
        released.bind(("127.0.0.1", ports[0]))
