from __future__ import annotations

import ctypes
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from .errors import BackendError


class NativeRpcServer:
    """Hosts llama.cpp's RPC backend directly from llama-cpp-python native libraries."""

    def __init__(self) -> None:
        self.endpoint: str | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._dll_directory: Any = None

    def start(self) -> str:
        if self.endpoint and self._thread and self._thread.is_alive():
            return self.endpoint
        port = _free_port()
        self.endpoint = f"127.0.0.1:{port}"
        self._error = None
        self._thread = threading.Thread(target=self._run, name="llama-rpc-worker", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self._error:
                raise BackendError("rpc_worker_failed", str(self._error))
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return self.endpoint
            except OSError:
                time.sleep(0.1)
        raise BackendError("rpc_worker_timeout", "Embedded llama.cpp RPC worker did not start.")

    def _run(self) -> None:
        try:
            import llama_cpp
            from llama_cpp import Llama
            from llama_cpp import llama_cpp as low

            if not getattr(Llama, "_Llama__backend_initialized", False):
                low.llama_backend_init()
                # Llama's high-level wrapper must not initialize the global backend twice.
                setattr(Llama, "_Llama__backend_initialized", True)

            lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
            if os.name == "nt":
                self._dll_directory = os.add_dll_directory(str(lib_dir))
            libraries = _load_native_libraries(lib_dir)

            load_all = _symbol(libraries, "ggml_backend_load_all_from_path", required=False)
            if load_all is not None:
                load_all.argtypes = [ctypes.c_char_p]
                load_all(str(lib_dir).encode())

            dev_count = _symbol(libraries, "ggml_backend_dev_count")
            dev_get = _symbol(libraries, "ggml_backend_dev_get")
            dev_type = _symbol(libraries, "ggml_backend_dev_type")
            start_server = _symbol(libraries, "ggml_backend_rpc_start_server")

            dev_count.restype = ctypes.c_size_t
            dev_get.argtypes = [ctypes.c_size_t]
            dev_get.restype = ctypes.c_void_p
            dev_type.argtypes = [ctypes.c_void_p]
            dev_type.restype = ctypes.c_int

            devices: list[int] = []
            for index in range(dev_count()):
                device = dev_get(index)
                # Current ggml enum: CPU=0, GPU=1, IGPU=2, ACCEL=3.
                if device and dev_type(device) in (1, 2, 3):
                    devices.append(device)
            if not devices:
                for index in range(dev_count()):
                    device = dev_get(index)
                    if device and dev_type(device) == 0:
                        devices.append(device)
                        break
            if not devices:
                raise RuntimeError("No llama.cpp backend device is available")

            array_type = ctypes.c_void_p * len(devices)
            device_array = array_type(*devices)
            start_server.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            start_server.restype = None
            start_server(
                (self.endpoint or "127.0.0.1:0").encode(),
                None,
                max(1, (os.cpu_count() or 8) // 2),
                len(devices),
                device_array,
            )
        except Exception as error:  # native bootstrap errors must reach the control plane
            self._error = error


def _load_native_libraries(directory: Path) -> list[Any]:
    patterns = ("*.dll", "*.so", "*.dylib")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(directory.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No llama.cpp native libraries were found under {directory}")
    libraries: list[Any] = []
    for path in paths:
        try:
            if os.name == "nt":
                libraries.append(ctypes.CDLL(str(path), winmode=0))
            else:
                libraries.append(ctypes.CDLL(str(path)))
        except OSError:
            continue
    if not libraries:
        raise RuntimeError(f"None of the llama.cpp native libraries under {directory} could be loaded")
    return libraries


def _symbol(libraries: list[Any], name: str, required: bool = True) -> Any:
    for library in libraries:
        try:
            return getattr(library, name)
        except AttributeError:
            continue
    if required:
        raise RuntimeError(
            f"{name} is unavailable. Rebuild llama-cpp-python with GGML_RPC=ON and GGML_CUDA=ON."
        )
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
