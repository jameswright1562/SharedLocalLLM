# SharedLocalLLM

**Use two Windows computers as one local LLM workspace.**

SharedLocalLLM finds GGUF models you already have, measures both computers and the network between
them, and recommends where each model should run. If a model benefits from—or requires—both GPUs,
the app can launch it through `llama.cpp`'s distributed RPC backend.

You do not need a particular GPU model or a matching pair of computers. Hardware, available memory,
model size, context length, and network performance are detected at runtime. LM Studio works as the
default model source, but it is optional: you can select any local model directory instead.

> [!IMPORTANT]
> **This project is an implementation preview.** The browser UI and automated frontend/native tests
> are working, but the full installer and physical two-computer GPU acceptance matrix are not yet
> complete. Do not read the current test results as a claim that distributed inference has been
> proven on real hardware. Both `llama.cpp` RPC and multimodal support are experimental.

## Why this project exists

A model may be too large for either computer on its own while still fitting across their combined
VRAM and system RAM. Starting the correct processes, choosing a safe layer split, checking whether
the network is fast enough, and keeping the RPC service off the open LAN are all easy to get wrong.

SharedLocalLLM puts that setup behind one desktop interface:

- Pair exactly two computers on a trusted Windows Private network by default, with an explicit
  temporary pairing override for a Public profile.
- Discover hardware and current free VRAM/RAM instead of relying on hardcoded specifications.
- Find LM Studio models automatically or index user-selected read-only folders.
- Recognize normal GGUF files, split GGUF models, and nearby `mmproj` vision projectors.
- Test throughput, latency, jitter, and packet loss in both directions.
- Compare single-computer, combined-GPU, and GPU-plus-RAM placements.
- Launch local chat and a localhost OpenAI-compatible API.
- Keep clear, redacted logs when a runtime, benchmark, or model fails.

Two computers increase available capacity, but they do not guarantee more speed. SharedLocalLLM
measures the available choices and can recommend one computer when that is faster.

## How it fits together

The computer that owns the selected model normally becomes the **coordinator**. It starts
`llama-server` and makes placement decisions. The other computer becomes the **worker** and starts
`ggml-rpc-server` on loopback only.

```text
Computer A                                              Computer B

Local app/API                                            Local app/API
127.0.0.1:11435                                          127.0.0.1:11435
      │                                                        │
      ▼                                                        │
SharedLocalLLM coordinator ◀── encrypted peer channel ──▶ SharedLocalLLM worker
      │                                                        │
      ▼                                                        ▼
llama-server                                       ggml-rpc-server on 127.0.0.1
      │                                                        │
      └──────────── model layers use both compute devices ─────┘
```

The raw RPC socket is **never meant to be exposed to the LAN**. Peer traffic uses an encrypted Noise
channel protected by an expiring six-digit pairing code. This is a trusted-private-LAN preview, not
production-hardened peer identity or authorization; stronger authentication is tracked in
[`ideas.md`](ideas.md). The local model API also stays on `127.0.0.1` and requires a bearer key.

Read [Architecture](docs/architecture.md) for component boundaries, lifecycle, persistence, and the
full security model.

## What you need

- Two Windows 10/11 x64 computers on the same trusted network. A Private profile is required to
  launch a cluster.
- An NVIDIA GPU and compatible driver on each computer for the pinned CUDA 12 runtime.
- A GGUF model stored on at least one computer.
- Enough combined VRAM and RAM for the chosen model and context.

There is no fixed VRAM, RAM, processor, Ethernet, or Wi-Fi specification. A wired network is usually
the best starting point, but the app measures the route rather than guessing from the adapter name.

## Two-computer setup

The intended installed flow is the same on both computers:

1. Install the same SharedLocalLLM version on both computers and open it on each one.
2. Install the pinned `llama.cpp` runtime from the first-run wizard. The runtime manager checks the
   official origin, archive size, SHA-256 digest, archive contents, and required executables.
3. Give each computer a friendly name.
4. Start pairing on one computer. Select the discovered peer—or enter its private IPv4 address—and
   confirm the six-digit code shown by both apps. If Windows classifies a trusted LAN as Public, the
   wizard can permit a temporary five-minute pairing session after a native warning. Showing a code
   also requires approval for an app-and-port-specific Public firewall rule, which is removed after
   pairing or timeout. The app still will not launch a cluster until the profile is Private or
   domain-authenticated.
5. Allow SharedLocalLLM through Windows Firewall for **Private networks only**.
6. Open **Models**. Use the detected LM Studio catalogue or choose **Add folder** for any other GGUF
   directory on either computer.
7. Open **Network**, run the bidirectional test, and review the result. Then select a model and inspect
   its fit/recommendation before launching it.
8. Use **Chat** from either computer, or copy the localhost API details from **API**. Stop the cluster
   from the app when finished so its managed processes are cleaned up.

There is not yet a published, physically validated installer release. Contributors can build the
current preview from source using the steps below. If pairing or launch fails, use the
[two-computer troubleshooting guide](docs/troubleshooting.md); every manual command there is clearly
labelled by computer.

## Models: LM Studio or any folder

Model discovery runs independently on both computers:

1. If LM Studio's `lms` command is available, the app reads `lms ls --json --detailed`. This follows
   the model location configured inside LM Studio, including a non-default location.
2. If the CLI is unavailable, it checks `%USERPROFILE%\.lmstudio\models`.
3. It also indexes every custom folder you add in SharedLocalLLM.

Configured folders are read-only. SharedLocalLLM does not move, rename, overwrite, delete, or copy
model files between computers. Keep split shards together. For a vision model, keep its compatible
`mmproj*.gguf` beside the model files.

LM Studio itself is not required. If it is running a loaded model and occupying substantial VRAM,
SharedLocalLLM reports the conflict and lets you choose whether to unload it; the app does not kill
LM Studio automatically.

## Network and placement recommendations

The first candidate split is based on currently usable VRAM, with safety space left for Windows and
other applications. The app can then compare valid nearby layer splits, single-computer placement,
and coordinator RAM spill. Results are tied to the exact model, context, hardware, drivers, runtime,
and active network adapter so a changed setup does not reuse a stale recommendation.

Automatic allocation is the default. In **Models**, switch to **Manual GPU split** to set a target
GPU-layer count for each connected computer. The live estimate combines proportional GGUF weights,
the selected context's F16 KV cache, and a runtime allowance, shows the current available VRAM on
each computer, and blocks a manual launch when the estimate exceeds it. `llama.cpp` may round the
requested proportions at tensor boundaries, so the displayed figures are estimates rather than a
guarantee of exact allocation.

The network rating is guidance, not a promise of token speed:

| Rating | Sustained bidirectional throughput |   p95 latency | Meaning                                  |
| ------ | ---------------------------------: | ------------: | ---------------------------------------- |
| Good   |                at least 800 Mbit/s |  at most 3 ms | Strong candidate for distributed testing |
| Usable |                at least 200 Mbit/s | at most 10 ms | Distribution is possible; benchmark it   |
| Poor   |      below either usable threshold |             — | Allowed with a clear performance warning |

## Try the project locally

### 1. Install the development tools

You will need:

- [Node.js](https://nodejs.org/) 22.12 or newer and pnpm 11.
- [Rust](https://rustup.rs/) stable with the `x86_64-pc-windows-msvc` target.
- Microsoft Visual Studio Build Tools with **Desktop development with C++**.
- Microsoft Edge WebView2 (normally already present on current Windows versions).

### 2. Install dependencies

From PowerShell in the repository root:

```powershell
corepack enable
pnpm install --frozen-lockfile
```

### 3. Choose a development mode

The browser preview is the quickest way to explore the interface. It uses simulated computers,
models, network results, and inference responses; it does not start the native distributed runtime.

```powershell
pnpm dev
```

Run the real Tauri desktop shell when working on native commands:

```powershell
pnpm tauri dev
```

### 4. Run the checks

```powershell
pnpm check
cargo fmt --manifest-path src-tauri/Cargo.toml --all --check
cargo test --manifest-path src-tauri/Cargo.toml --all-targets
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
```

The frontend coverage gate is 80% for statements, branches, functions, and lines. Playwright covers
the critical browser-preview journeys. Native tests cover deterministic core and peer behavior, but
neither replaces the physical two-computer checks in [Testing](docs/testing.md).

### 5. Build the Windows installer

```powershell
pnpm tauri build --bundles nsis
```

The installer is written below `src-tauri/target/release/bundle/nsis/`. Local builds are unsigned
unless you configure a Windows code-signing certificate, so Windows may show its usual warning.

## Local API example

Each computer uses `http://127.0.0.1:11435` by default. The app does not silently switch ports when
that address is occupied. Copy the current URL and key from the **API** page, then call it from the
same computer:

```powershell
$apiKey = "paste-the-key-shown-by-SharedLocalLLM"
curl.exe http://127.0.0.1:11435/v1/chat/completions `
  -H "Authorization: Bearer $apiKey" `
  -H "Content-Type: application/json" `
  -d '{"model":"selected-model","stream":false,"messages":[{"role":"user","content":"Hello"}]}'
```

Baseline routes are `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions`, including
SSE streaming. Keep the bearer key out of logs, screenshots, and issue reports.

## Safety and current limits

- Use only two computers you control on a trusted Windows Private network. Do not expose this app to
  the internet, an untrusted LAN, or a multi-tenant environment.
- V1 supports Windows x64 NVIDIA computers and layer splitting. WAN clustering, more than two nodes,
  and LAN tensor parallelism are out of scope.
- Upstream describes `llama.cpp` RPC as proof-of-concept, fragile, and insecure. SharedLocalLLM keeps
  raw RPC on loopback, but the current peer-authentication design is still a preview.
- Distributed inference may be slower than the best single computer, especially over Wi-Fi or when
  spilling layers into system RAM.
- A generation cannot migrate if the peer disconnects. It returns an error and may offer a
  single-computer retry only when the model fits.
- GGUF models are not sandboxed. Use models from sources you trust.
- Vision compatibility depends on the model and projector and remains experimental.
- `llama.cpp` filesystem, shell, MCP, and built-in agent tools are never enabled.

See the upstream [`llama.cpp` RPC warning](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
and [security guidance](https://github.com/ggml-org/llama.cpp/security) for the risks behind these
boundaries.

## Project guide

| If you want to…                                        | Read…                                                   |
| ------------------------------------------------------ | ------------------------------------------------------- |
| Understand the processes, protocol, and state          | [Architecture](docs/architecture.md)                    |
| Run checks or see the physical acceptance matrix       | [Testing and release gates](docs/testing.md)            |
| Fix discovery, firewall, network, model, or API issues | [Two-computer troubleshooting](docs/troubleshooting.md) |
| Contribute code safely                                 | [Contributing](CONTRIBUTING.md)                         |
| Review deferred improvements                           | [Ideas](ideas.md)                                       |
| Report a security concern                              | [Security policy](SECURITY.md)                          |

## Contributing

New contributors are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), keep changes focused,
and add or update a test before changing behavior. Please report the exact checks you ran and call
out anything that still needs real-hardware validation.

## License

SharedLocalLLM is available under the [MIT License](LICENSE). Downloaded or bundled third-party
components keep their own licenses.
