# SharedLocalLLM Python backend

This branch moves application orchestration, peer networking, model discovery, benchmarking, and
`llama.cpp` inference into Python. Tauri remains a thin Windows shell and request bridge.

The backend uses `llama-cpp-python==0.3.35`. Install it with both CUDA and RPC enabled:

```powershell
pnpm backend:install
```

For local development the Rust shell starts `backend/.venv/Scripts/python.exe -m sharedlocalllm_backend`
and (in debug builds) auto-restarts it when the package sources change. The packaged app uses the
PyInstaller sidecar created by `pnpm backend:package`.

Ports remain loopback/private by design:

- `127.0.0.1:11436` - internal Tauri-to-Python control API
- configured `127.0.0.1:11435` by default - OpenAI-compatible API
- TCP `49158` - SharedLocalLLM peer control/tunnel transport
- UDP `49157` - LAN discovery announcements
- raw llama.cpp RPC - dynamically allocated loopback endpoints only

The native RPC worker is hosted directly from the `ggml-base` / `ggml-rpc` libraries bundled by the
RPC-enabled `llama-cpp-python` build; it is not exposed to the LAN.
