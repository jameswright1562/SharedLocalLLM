from __future__ import annotations

import ctypes
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from .errors import BackendError

# llama.cpp device types (ggml-backend.h)
GGML_DEVICE_CPU = 0
GGML_DEVICE_GPU = 1
GGML_DEVICE_IGPU = 2
GGML_DEVICE_ACCEL = 3

_native_libraries: list[Any] | None = None
_lib_directory: Path | None = None
_backend_initialized = False
_patch_installed = False
_rpc_devices_by_endpoint: dict[str, list[dict[str, Any]]] = {}
# Device list to apply to the next llama_model_params. llama.cpp reads
# params.devices only while loading the model, so a single pending slot is
# enough; every Llama construction in this process goes through _load_sync.
_pending_devices: list[list[int] | None] = []


class NativeRpcServer:
    """Hosts llama.cpp's RPC backend directly from llama-cpp-python native libraries."""

    def __init__(self) -> None:
        self.endpoint: str | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._dll_directory: Any = None
        self.include_cpu = True

    def start(self, include_cpu: bool = False) -> str:
        if (
            self.endpoint
            and self._thread
            and self._thread.is_alive()
        ):
            return self.endpoint
        # Expose the stable superset once. The coordinator omits the final CPU
        # device unless remote CPU offload is explicitly selected. llama.cpp's
        # embedded RPC server has no shutdown API, so starting a fresh daemon on
        # every toggle would leak threads and ports for the process lifetime.
        self.include_cpu = True
        self.endpoint = None
        self._error = None
        port = _free_port()
        self.endpoint = f"127.0.0.1:{port}"
        self._thread = threading.Thread(
            target=self._run, args=(True,), name="llama-rpc-worker", daemon=True
        )
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

    def _run(self, include_cpu: bool) -> None:
        try:
            ensure_backend_initialized()

            dev_count = _symbol("ggml_backend_dev_count")
            dev_get = _symbol("ggml_backend_dev_get")
            dev_type = _symbol("ggml_backend_dev_type")
            start_server = _symbol("ggml_backend_rpc_start_server")

            dev_count.restype = ctypes.c_size_t
            dev_get.argtypes = [ctypes.c_size_t]
            dev_get.restype = ctypes.c_void_p
            dev_type.argtypes = [ctypes.c_void_p]
            dev_type.restype = ctypes.c_int

            # GPU-class devices first, then the CPU device last. The fixed order is
            # part of the RPC contract: the coordinator identifies the remote CPU as
            # the final RPC device because llama.cpp reports every RPC device as a
            # GPU regardless of its real type.
            gpu_devices: list[int] = []
            cpu_device: int | None = None
            for index in range(dev_count()):
                device = dev_get(index)
                # Current ggml enum: CPU=0, GPU=1, IGPU=2, ACCEL=3.
                if device and dev_type(device) in (1, 2, 3):
                    gpu_devices.append(device)
                elif device and dev_type(device) == 0 and cpu_device is None:
                    cpu_device = device
            if gpu_devices:
                devices = gpu_devices + ([cpu_device] if include_cpu and cpu_device else [])
            elif cpu_device:
                # No GPU present: keep the historical no-GPU fallback to the CPU.
                devices = [cpu_device]
            else:
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


def prepare_rpc_load(
    endpoint: str | None, remote_gpu_layers: int, remote_cpu_layers: int, local_layers: int
) -> list[float] | None:
    """Register the worker RPC device and stage the device list + tensor split.

    Called on the coordinator before constructing Llama. Registers the remote
    worker (when ``endpoint`` is set), restricts the model to exactly the
    devices we intend to use (so stale RPC devices from earlier sessions never
    leak into a local load), and returns the tensor_split for the registered
    device order. Returns ``None`` when running without a remote worker.

    ``remote_cpu_layers`` gates whether the worker's CPU is treated as an
    offload target; when zero, the CPU device is not part of the model device
    list and every RPC device is treated as a GPU (the historical behaviour).
    """
    ensure_backend_initialized()
    selected_rpc: list[dict[str, Any]] = []
    if endpoint:
        selected_rpc = _rpc_devices_by_endpoint.get(endpoint, [])
        if not selected_rpc:
            before = _rpc_device_count()
            register_rpc_client(endpoint)
            fresh_count = _rpc_device_count() - before
            info = device_info()
            rpc, _ = classify_devices(info)
            selected_rpc = rpc[-fresh_count:] if fresh_count else []
            if not selected_rpc:
                raise BackendError(
                    "rpc_device_missing",
                    "The worker connected, but llama.cpp did not register an RPC device.",
                )
            _rpc_devices_by_endpoint[endpoint] = selected_rpc
    info = device_info()
    rpc, gpus = classify_devices(info)
    del rpc
    remote_gpus, remote_cpus = split_rpc_devices(
        selected_rpc,
        include_cpu=True,
    )
    if remote_gpu_layers > 0 and not remote_gpus:
        raise BackendError(
            "remote_gpu_missing",
            "GPU layers were assigned to a worker that did not expose a GPU device.",
        )
    devices: list[int] = []
    if endpoint:
        # llama.cpp orders RPC devices before local GPUs; keep that order. The
        # remote CPU is the final RPC device and is added only when layers are
        # explicitly assigned to it.
        devices.extend(value["pointer"] for value in remote_gpus)
        if remote_cpu_layers > 0:
            devices.extend(value["pointer"] for value in remote_cpus)
    devices.extend(value["pointer"] for value in gpus)
    prepare_model_devices(devices)
    if not endpoint:
        return None
    return tensor_split_for(
        len(remote_gpus), len(remote_cpus) if remote_cpu_layers > 0 else 0, len(gpus),
        remote_gpu_layers, remote_cpu_layers, local_layers,
    )


def _rpc_device_count() -> int:
    """Count RPC backend devices exactly as ``classify_devices`` sees them."""
    return len(classify_devices(device_info())[0])


def classify_devices(
    info: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split device snapshots into RPC servers and local GPUs (iGPU fallback)."""
    rpc = [
        value for value in info
        if value["type"] == GGML_DEVICE_GPU and value["registry"] == "RPC"
    ]
    gpus = [
        value for value in info
        if value["type"] == GGML_DEVICE_GPU and value["registry"] != "RPC"
    ]
    if not gpus:
        gpus = [value for value in info if value["type"] == GGML_DEVICE_IGPU]
    return rpc, gpus


def split_rpc_devices(
    rpc: list[dict[str, Any]], include_cpu: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split RPC devices into remote GPUs and the remote CPU device.

    The worker exposes its devices in a fixed order (GPU-class devices first,
    the CPU last). llama.cpp reports every RPC device as GPU type on the
    coordinator, so that ordering is the only reliable way to identify the
    remote CPU. When ``include_cpu`` is false the whole list is treated as
    GPUs, preserving the historical no-GPU fallback behaviour.
    """
    if include_cpu and rpc:
        return rpc[:-1], rpc[-1:]
    return rpc, []


def tensor_split_for(
    rpc_gpu_count: int, rpc_cpu_count: int, gpu_count: int,
    remote_gpu_layers: int, remote_cpu_layers: int, local_layers: int,
) -> list[float]:
    """Build the tensor_split in llama.cpp device order (RPC GPUs, RPC CPU, local GPUs)."""
    split: list[float] = []
    if rpc_gpu_count > 0:
        split.extend([remote_gpu_layers / rpc_gpu_count] * rpc_gpu_count)
    if rpc_cpu_count > 0:
        split.extend([remote_cpu_layers / rpc_cpu_count] * rpc_cpu_count)
    if gpu_count > 0:
        split.extend([local_layers / gpu_count] * gpu_count)
    return split


def register_rpc_client(endpoint: str) -> None:
    """Register a remote llama.cpp RPC worker as a local backend device."""
    ensure_backend_initialized()
    add_server = _symbol("ggml_backend_rpc_add_server")
    register = _symbol("ggml_backend_register")
    add_server.argtypes = [ctypes.c_char_p]
    add_server.restype = ctypes.c_void_p
    register.argtypes = [ctypes.c_void_p]
    register.restype = None

    registry = add_server(endpoint.encode())
    if not registry:
        raise BackendError(
            "rpc_server_unavailable",
            f"The worker on the other computer did not answer the RPC handshake at {endpoint}.",
            "Check that both computers run the same build and that the private link is up.",
        )
    register(registry)


def prepare_model_devices(devices: list[int] | None) -> None:
    """Declare the device list for the next Llama construction."""
    _pending_devices[:] = [devices]


def device_info() -> list[dict[str, Any]]:
    """Snapshot the currently registered ggml devices for split planning."""
    ensure_backend_initialized()
    dev_count = _symbol("ggml_backend_dev_count")
    dev_get = _symbol("ggml_backend_dev_get")
    dev_name = _symbol("ggml_backend_dev_name")
    dev_type = _symbol("ggml_backend_dev_type")
    dev_reg = _symbol("ggml_backend_dev_backend_reg")
    reg_name = _symbol("ggml_backend_reg_name")

    dev_count.restype = ctypes.c_size_t
    dev_get.argtypes = [ctypes.c_size_t]
    dev_get.restype = ctypes.c_void_p
    dev_name.argtypes = [ctypes.c_void_p]
    dev_name.restype = ctypes.c_char_p
    dev_type.argtypes = [ctypes.c_void_p]
    dev_type.restype = ctypes.c_int
    dev_reg.argtypes = [ctypes.c_void_p]
    dev_reg.restype = ctypes.c_void_p
    reg_name.argtypes = [ctypes.c_void_p]
    reg_name.restype = ctypes.c_char_p

    values: list[dict[str, Any]] = []
    for index in range(dev_count()):
        device = dev_get(index)
        if not device:
            continue
        registry = dev_reg(device)
        values.append({
            "name": _text(dev_name(device)),
            "type": dev_type(device),
            "registry": _text(reg_name(registry)) if registry else "",
            "pointer": device,
        })
    return values


def ensure_backend_initialized() -> None:
    """Initialize llama.cpp's global backend exactly once."""
    global _backend_initialized
    if _backend_initialized:
        return
    from llama_cpp import Llama
    from llama_cpp import llama_cpp as low

    if not getattr(Llama, "_Llama__backend_initialized", False):
        low.llama_backend_init()
        # Llama's high-level wrapper must not initialize the global backend twice.
        setattr(Llama, "_Llama__backend_initialized", True)
    load_all = _symbol("ggml_backend_load_all_from_path", required=False)
    if load_all is not None:
        load_all.argtypes = [ctypes.c_char_p]
        load_all(str(_libraries_dir()).encode())
    _install_model_devices_patch()
    _backend_initialized = True


def runtime_health() -> dict[str, Any]:
    try:
        import llama_cpp

        ensure_backend_initialized()
        _symbol("ggml_backend_rpc_start_server")
        _symbol("ggml_backend_rpc_add_server")
        return {"status": "ready", "version": f"llama-cpp-python {llama_cpp.__version__}"}
    except Exception as error:
        return {
            "status": "error",
            "version": None,
            "error": f"llama.cpp backend health check failed: {error}",
        }


def _install_model_devices_patch() -> None:
    """Let LlamaModel receive the pending device list before loading.

    llama-cpp-python's Llama wrapper ignores params.devices, but llama.cpp uses
    it to select exactly which backend devices a model may offload to. Inject it
    at the boundary where the native model is loaded. This is a private API,
    stable because llama-cpp-python is pinned in pyproject.toml.
    """
    global _patch_installed
    if _patch_installed:
        return
    import llama_cpp._internals as internals

    original = internals.LlamaModel.__init__

    def patched(
        self: Any, *, path_model: str, params: Any, verbose: bool = True
    ) -> None:
        pending = _pending_devices.pop() if _pending_devices else None
        if pending is not None:
            # Keep the array alive for the life of the model; llama.cpp copies
            # the struct (including this pointer) into the loaded model.
            self._c_devices = _device_array(pending)
            params.devices = ctypes.cast(self._c_devices, ctypes.c_void_p)
        original(self, path_model=path_model, params=params, verbose=verbose)

    internals.LlamaModel.__init__ = patched
    _patch_installed = True


def _device_array(devices: list[int]) -> Any:
    """Build a NULL-terminated ggml_backend_dev_t array from pointer values."""
    array = (ctypes.c_void_p * (len(devices) + 1))()
    for index, pointer in enumerate(devices):
        array[index] = pointer
    return array


def _libraries_dir() -> Path:
    global _lib_directory
    if _lib_directory is None:
        import llama_cpp
        _lib_directory = Path(llama_cpp.__file__).resolve().parent / "lib"
        if os.name == "nt":
            os.add_dll_directory(str(_lib_directory))
    return _lib_directory


def _libraries() -> list[Any]:
    global _native_libraries
    if _native_libraries is None:
        _native_libraries = _load_native_libraries(_libraries_dir())
    return _native_libraries


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


def _symbol(name: str, required: bool = True) -> Any:
    for library in _libraries():
        try:
            return getattr(library, name)
        except AttributeError:
            continue
    if required:
        raise RuntimeError(
            f"{name} is unavailable. Rebuild llama-cpp-python with GGML_RPC=ON and GGML_CUDA=ON."
        )
    return None


def _text(value: bytes | None) -> str:
    return value.decode("utf-8", "replace") if value else ""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
