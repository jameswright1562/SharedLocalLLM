from __future__ import annotations

import asyncio

import uvicorn

from .api import CONTROL_PORT, ApiServerManager, create_control_app
from .runtime import BackendRuntime


async def run_backend() -> None:
    runtime = BackendRuntime()
    api = ApiServerManager(runtime)
    runtime.api_port_changed = api.restart
    await api.start(int(runtime.store.get("apiPort", 11435)))
    control = uvicorn.Server(uvicorn.Config(
        create_control_app(runtime), host="127.0.0.1", port=CONTROL_PORT, log_level="info"
    ))
    try:
        await control.serve()
    finally:
        await api.stop()


def main() -> None:
    asyncio.run(run_backend())


if __name__ == "__main__":
    main()
