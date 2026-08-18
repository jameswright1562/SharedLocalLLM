from __future__ import annotations

import uvicorn

from .api import CONTROL_PORT, ApiServerManager, create_control_app
from .runtime import BackendRuntime


def main() -> None:
    runtime = BackendRuntime()
    api = ApiServerManager(runtime)
    runtime.api_port_changed = api.restart
    api.start(int(runtime.store.get("apiPort", 11435)))
    try:
        uvicorn.run(create_control_app(runtime), host="127.0.0.1", port=CONTROL_PORT, log_level="info")
    finally:
        api.stop()


if __name__ == "__main__":
    main()
