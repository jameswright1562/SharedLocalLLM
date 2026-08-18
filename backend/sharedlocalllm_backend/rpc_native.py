from __future__ import annotations

import ctypes
import os
import socket
import threading
import time
from pathlib import Path

from .errors import BackendError


class NativeRpcServer:
    """Hosts llama.cpp's RPC backend directly from llama-cpp-python native libraries."""

    def __init__(self) -> None:
        self.endpoint: str | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> str:
        if self.endpoint and self._thread and self._thread.is_alive():
            return self.endpoint
        port = _free_port()
        self.endpoint = f"127.0.0.1:{port}"
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
            from llama_cpp import llama_cpp as low

            low.llama_backend_init()
            lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
            if os.name == "nt":
                os.add_dll_directory(str(lib_dir))
            base = ctypes.CDLL(str(_find_library(lib_dir, "ggml-base")), winmode=0 if os.name == "nt" else None)
            rpc = ctypes.CDLL(str(_find_library(lib_dir, "ggml-rpc")), winmode=0 if os.name == "nt" else None)
            base.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
            base.ggml_backend_load_all_from_path(str(lib_dir).encode())
            base.ggml_backend_dev_count.restype = ctypes.c_size_t
            base.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
            base.ggml_backend_dev_get.restype = ctypes.c_void_p
            base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
            base.ggml_backend_dev_type.restype = ctypes.c_int
            devices: list[int] = []
            for index in range(base.ggml_backend_dev_count()):
                device = base.ggml_backend_dev_get(index)
                if device and base.ggml_backend_dev_type(device) in (1, 2):
                    devices.append(device)
            if not devices:
                for index in range(base.ggml_backend_dev_count()):
                    device = base.ggml_backend_dev_get(index)
                    if device and base.ggml_backend_dev_type(device) == 0:
                        devices.append(device)
                        break
            if not devices:
                raise RuntimeError("No llama.cpp backend device is available")
            array_type = ctypes.c_void_p * len(devices)
            device_array = array_type(*devices)
            rpc.ggml_backend_rpc_start_server.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            endpoint = (self.endpoint or "127.0.0.1:0").encode()
            rpc.ggml_backend_rpc_start_server(
                endpoint, None, max(1, (os.cpu_count() or 8) // 2), len(devices), device_array
            )
        except Exception as error:  # native bootstrap errors must reach the control plane
            self._error = error


def _find_library(directory: Path, stem: str) -> Path:
    suffixes = ("*.dll", "*.so", "*.dylib")
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend(directory.glob(f"*{stem}*{suffix[1:]}"))
    if not candidates:
        raise FileNotFoundError(f"{stem} native library was not found under {directory}")
    return candidates[0]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
