# SharedLocalLLM

**Use two Windows computers as one local LLM workspace.**

This branch uses a React/Tauri desktop shell with a Python backend. Python owns model discovery,
peer networking, distributed RPC orchestration, inference, benchmarking, and the localhost
OpenAI-compatible API. Rust is intentionally thin and handles Windows/Tauri integration such as the
tray, autostart, firewall elevation, folder picking, and starting/stopping the packaged backend.

## Architecture

```text
Computer A                                                    Computer B

React UI                                                       React UI
   │                                                              │
Tauri/Rust                                                     Tauri/Rust
   │                                                              │
Python backend                                                Python backend
   │                                                              │
   ├── FastAPI / OpenAI API                                      ├── peer server
   ├── asyncio peer client/server                                └── embedded llama.cpp RPC worker
   ├── llama-cpp-python                                                  │
   │      │                                                            GPU
   │      ├── local CUDA GPU                                            │
   │      └── rpc_servers=127.0.0.1:<forwarder>                         │
   │                         │                                           │
   └── loopback RPC forwarder ───── TCP 49158 peer tunnel ──────────────┘
```

Raw llama.cpp RPC remains loopback-only. TCP `49158` carries the SharedLocalLLM control/RPC tunnel
between the two computers and UDP `49157` is used for discovery. The peer protocol is versioned but
is currently intended only for two trusted computers on a LAN you control.

See [Architecture](docs/architecture.md) for the component and lifecycle details.

## Backend stack

- Python 3.12+
- `llama-cpp-python==0.3.34`, built with `GGML_CUDA=ON` and `GGML_RPC=ON`
- `asyncio` for peer control, discovery, network testing, and byte-stream forwarding
- FastAPI + Uvicorn for the internal bridge and OpenAI-compatible API
- psutil + `nvidia-smi` for hardware telemetry
- PyInstaller for the packaged Windows backend executable

`llama-cpp-python` exposes `rpc_servers`, `tensor_split`, `split_mode`, and `n_gpu_layers` directly,
so the application no longer needs a Rust wrapper around llama.cpp just to configure distributed
inference.

## Ports

| Port                   | Scope    | Purpose                                      |
| ---------------------- | -------- | -------------------------------------------- |
| `11436/tcp`            | loopback | Internal Tauri-to-Python control API         |
| `11435/tcp` by default | loopback | OpenAI-compatible API                        |
| `49158/tcp`            | LAN      | Peer control and tunneled llama.cpp RPC      |
| `49157/udp`            | LAN      | Peer discovery                               |
| dynamic TCP            | loopback | Raw llama.cpp RPC worker/forwarder endpoints |

The OpenAI API requires the bearer key shown by the app. The Python migration generates a new API
key the first time it runs because the previous key was stored using the Rust/Windows DPAPI path.
Non-secret settings and the existing peer record are migrated from `%LOCALAPPDATA%\SharedLocalLLM\settings.json`.

## Development setup

Install Node.js 22.12+, pnpm 11, Rust/MSVC, Python 3.12, CMake, and an NVIDIA CUDA toolkit. Then:

```powershell
corepack enable
pnpm install --frozen-lockfile
pnpm backend:install
pnpm tauri dev
```

`pnpm backend:install` creates `backend/.venv`, installs Ninja, and builds llama-cpp-python from source
with CUDA and RPC enabled. `pnpm tauri dev` checks that the backend environment is present before
starting Vite/Tauri.

The browser-only preview still uses the deterministic demo service:

```powershell
pnpm dev
```

## Checks

```powershell
pnpm typecheck
pnpm lint
pnpm test
pnpm backend:test
cargo fmt --manifest-path src-tauri/Cargo.toml --all --check
cargo test --manifest-path src-tauri/Cargo.toml --all-targets
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
```

## Build the Windows installer

```powershell
pnpm tauri build --bundles nsis
```

The Tauri build first packages the Python backend with PyInstaller and then bundles
`sharedlocalllm-backend.exe` as an application resource. End users do not need to install Python.

## Model discovery and placement

SharedLocalLLM scans the default LM Studio locations plus folders added through the UI. GGUF files
remain read-only and are never copied between machines. Model metadata is used to estimate context
and GPU placement. Remote catalogue entries can be launched from either UI: the computer that owns
the GGUF becomes coordinator, while a distributed local model can offload layers through the peer's
embedded RPC worker.

For two GPUs, the UI's node allocations are translated into llama.cpp `tensor_split` weights using
explicit node identity. Registered RPC devices are accounted for before the local CUDA device so an
asymmetric split is not accidentally reversed.

## Local OpenAI API

The default endpoint is `http://127.0.0.1:11435` and supports:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- SSE chat streaming when the local computer is coordinator

Example:

```powershell
$apiKey = "paste-the-key-shown-by-SharedLocalLLM"
curl.exe http://127.0.0.1:11435/v1/chat/completions `
  -H "Authorization: Bearer $apiKey" `
  -H "Content-Type: application/json" `
  -d '{"model":"active","stream":false,"messages":[{"role":"user","content":"Hello"}]}'
```

## Current limits

- The physical two-PC CUDA/RPC path still needs to be exercised on real hardware after checkout.
- Vision attachments are not wired into the Python inference handler yet. The `use-new-crate` branch
  also had text-only generation despite retaining projector metadata, so this branch does not remove
  a working multimodal generation path.
- The peer protocol is not encrypted or authenticated and should only be used on a trusted LAN.
- Distributed inference can be slower than single-GPU inference; benchmark the actual model/context.
- More than two nodes and WAN clustering are out of scope.

Both computers must run the Python-backend branch because its peer protocol version intentionally
differs from the Rust-backend branch.
