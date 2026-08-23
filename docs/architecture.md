# Architecture

SharedLocalLLM runs the same application on both Windows computers. Roles are selected at runtime.
The desktop process is deliberately split into a small native shell and a Python control/inference
backend.

## Components

### React renderer

Owns setup, nodes, model selection, placement controls, network measurements, chat, API details, and
diagnostics. Its existing `AppService` interface is preserved.

### Thin Tauri/Rust shell

Rust no longer owns inference or peer networking. It is responsible for:

- starting and supervising the Python backend process;
- bridging Tauri commands to the backend's loopback control endpoint;
- Windows tray/window lifecycle;
- Start-with-Windows registration;
- folder selection and opening Windows Settings/log folders;
- UAC and Windows Firewall rules for the Python-owned peer ports.

The shell does not link `llama-cpp-4` or `llama-cpp-sys-4`.

### Python backend

`backend/sharedlocalllm_backend` owns:

- persisted application state and migration from the previous Rust settings file;
- hardware telemetry and GGUF/LM Studio discovery;
- peer discovery, connection, heartbeat, network benchmark, and remote catalogue;
- RPC forwarding over the peer TCP channel;
- `llama-cpp-python` model loading and generation;
- embedded llama.cpp RPC worker hosting through the native libraries shipped with llama-cpp-python;
- inference benchmarks;
- local OpenAI-compatible HTTP endpoints.

The backend's internal control server binds `127.0.0.1:11436`. It is an implementation detail and is
only called by the local Tauri shell.

## Distributed model load

```text
Coordinator                                        Worker

start_cluster
    │
    ├─ resolve model + node allocation
    ├─ start loopback RpcForwarder
    │       │
    │       └──────── TCP 49158 / rpc_tunnel ──────────┐
    │                                                   │
    └─ Llama(                                           ▼
         model_path=...,                        embedded ggml RPC server
         n_gpu_layers=...,                             │
         split_mode=LAYER,                             ▼
         tensor_split=[remote, local],             worker CUDA GPU
         rpc_servers="127.0.0.1:<forwarder>"
       )
         │
         └─ local CUDA GPU
```

The raw RPC server and coordinator-side RPC forwarder both bind dynamically allocated loopback
ports. Only the SharedLocalLLM peer listener is visible to the LAN.

llama.cpp registers RPC devices before local CUDA devices. The Python inference layer therefore
constructs `tensor_split` in remote-then-local order rather than assuming the UI order equals the
native device order.

## Peer protocol

UDP `49157` announces the app on each IPv4 interface. The broadcaster binds each usable interface
before sending to its directed broadcast address, which lets a dedicated direct-Ethernet link be
used even when Wi-Fi remains the default Internet route.

TCP `49158` carries newline-delimited JSON control requests. An `rpc_tunnel` request changes that
connection into a raw bidirectional byte tunnel after a short ready response. Protocol version 5 is
specific to the Python backend branch, so mixed Rust/Python peers fail explicitly instead of trying
to interpret different framing.

Connect is symmetric: the initiator sends its node capabilities and local model catalogue, and the
receiver persists the initiator using the source IPv4 address. Both computers therefore have enough
state to reconnect after a restart.

## Inference concurrency

There is one `InferenceEngine` and one loaded `llama_cpp.Llama` instance per backend process.
Operations are serialized with an asyncio lock and a native-thread lock. Blocking llama.cpp calls
run off the event loop. The control API, OpenAI API, peer server, and inference coordinator all run
on the same asyncio event loop so an asyncio primitive is never shared across event loops.

OpenAI SSE generation uses llama-cpp-python's streaming iterator on a worker thread and forwards
chunks through an asyncio queue. The Tauri renderer currently receives the completed response via
the generic command bridge; the public OpenAI endpoint streams tokens when this computer owns the
model.

## Worker RPC lifecycle

The worker starts lazily on the first RPC tunnel request. Python loads the native libraries bundled
with llama-cpp-python, locates the exported ggml backend registry/RPC symbols, enumerates accelerator
devices, and calls `ggml_backend_rpc_start_server` on a daemon OS thread. The server is never bound
to a LAN address.

The llama.cpp global backend is initialized exactly once. The code synchronizes that initialization
with llama-cpp-python's high-level `Llama` class so later model loads do not initialize the same
global backend twice.

## HTTP APIs

- `127.0.0.1:11436`: internal Tauri control bridge.
- configured `127.0.0.1:11435` by default: bearer-protected OpenAI-compatible API.

The API port can be changed at runtime. The replacement port is checked before the current server is
stopped, then Uvicorn is restarted on the same asyncio loop.

## Persistence

Python writes `%LOCALAPPDATA%\SharedLocalLLM\python-backend.json`. On first use it imports compatible
fields from the previous `settings.json`, including install identity, device name, custom model
folders, peer record, autostart preference, and benchmark history. The previous DPAPI API key is not
read by Python; a new local bearer key is generated for this branch.

A successful `start_cluster` also records the model's normalized load configuration under
`modelLoadConfigs`: context size, GPU layer split, remote-CPU offload, force flag, and the advanced
options (flash attention, mmap, mlock, CPU threads, batch size). The snapshot exposes these entries
and the renderer pre-fills the model inspector with them, so relaunching a model reuses the exact
values that launched it previously. A failed load never replaces the saved entry, and temporary
loads for inference benchmarks skip saving so they cannot overwrite the user's configuration.

## Windows firewall

The Python process owns the peer sockets, so old program-scoped rules for the Rust executable cannot
be reused. The shell creates two new Profile-Any, port-scoped rules:

- `SharedLocalLLM Peer Backend` — TCP 49158
- `SharedLocalLLM Peer Discovery` — UDP 49157

This avoids the Windows Public/Private profile distinction and works for direct Ethernet links.

## Packaging

Development uses `backend/.venv/Scripts/python.exe`. Production runs a PyInstaller executable that
Tauri bundles as `backend/sharedlocalllm-backend.exe`. `llama-cpp-python` is built from source with
CUDA, RPC, and shared native libraries enabled before packaging.

The installed application therefore does not require a user-managed Python installation.
