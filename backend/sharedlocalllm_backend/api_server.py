"""Loopback API socket ownership and startup port selection."""

from __future__ import annotations

import asyncio
import errno
import socket
from typing import Any, Protocol

import uvicorn

from .api import CONTROL_PORT, create_openai_app
from .errors import BackendError


class ApiStarter(Protocol):
    async def start(self, port: int) -> None: ...


def _bind_loopback(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(2048)
        sock.setblocking(False)
        return sock
    except BaseException:
        sock.close()
        raise


class ApiServerManager:
    """Owns a bound socket before handing it to Uvicorn on the control loop."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.server: uvicorn.Server | None = None
        self.task: asyncio.Task[None] | None = None
        self.port: int | None = None
        self._socket: socket.socket | None = None

    async def start(self, port: int) -> None:
        if self.is_healthy() and self.port == port:
            return
        await self.stop()
        # Binding here makes conflicts ordinary OSErrors, and reserves the port
        # atomically. Uvicorn's own bind path would sys.exit from its server task.
        self._socket = _bind_loopback(port)
        self.port = int(self._socket.getsockname()[1])
        try:
            config = uvicorn.Config(
                create_openai_app(self.runtime), host="127.0.0.1", port=self.port,
                log_level="warning", access_log=False, lifespan="off",
            )
            self.server = uvicorn.Server(config)
            self.task = asyncio.create_task(self.server.serve(sockets=[self._socket]))
            for _ in range(100):
                if self.server.started:
                    return
                if self.task.done():
                    await self.task
                    raise BackendError("api_start_failed", f"OpenAI API failed to start on port {port}.")
                await asyncio.sleep(0.05)
            raise BackendError("api_start_timeout", f"OpenAI API did not start on port {port} in time.")
        except BaseException:
            await self.stop()
            raise

    async def restart(self, port: int) -> None:
        previous = self.port
        try:
            await self.start(port)
        except Exception:
            if previous is not None and previous != port:
                await self.start(previous)
            raise

    def is_healthy(self) -> bool:
        return bool(self.server and self.server.started and self.task and not self.task.done())

    async def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
        try:
            if self.task:
                try:
                    await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), 3)
                except asyncio.TimeoutError:
                    self.task.cancel()
                    await asyncio.gather(self.task, return_exceptions=True)
        finally:
            if self._socket:
                self._socket.close()
            self._socket = None
            self.server = None
            self.task = None
            self.port = None


async def start_with_port_fallback(
    manager: ApiStarter, store: Any, preferred_port: int, attempts: int = 16,
) -> int:
    """Try successive loopback ports, excluding the fixed control port.

    Only binding conflicts trigger fallback. The winning port is persisted so
    snapshots, the API page, and client examples follow the actual listener.
    """
    if not 1024 <= preferred_port <= 65535 or attempts < 1:
        raise BackendError("api_port_invalid", "Use an API port from 1024-65535 and at least one attempt.")
    last_error: OSError | None = None
    port = preferred_port
    tried = 0
    while tried < attempts and port <= 65535:
        if port == CONTROL_PORT:
            port += 1
            continue
        tried += 1
        try:
            await manager.start(port)
        except OSError as error:
            if error.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            last_error = error
            port += 1
            continue
        if port != preferred_port:
            store.update(apiPort=port)
            store.log(
                "WARN", "api_port_auto_picked",
                f"127.0.0.1:{preferred_port} was unavailable; the API moved to 127.0.0.1:{port}.",
            )
        return port
    raise BackendError(
        "api_start_failed", f"No available loopback API port was found after {tried} attempts.",
        "Choose another API port or close the application using it.",
    ) from last_error
