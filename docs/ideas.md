# SharedLocalLLM ideas

This file records worthwhile follow-up work intentionally outside the current v1 delivery gate.

## Improved peer authentication and authorization

The Python-preview peer channel is intended only for two computers controlled by the same person on a trusted LAN. It is currently plain, unauthenticated TCP and keeps raw `llama.cpp` RPC plus the model API on loopback. This is adequate for functional preview testing but is not the desired long-term trust model.

Future authentication work should include:

- Use the six-digit code only to authenticate a one-time PAKE or key exchange; never reuse the code itself as durable key material.
- Generate a fresh high-entropy peer key during pairing and protect it with Windows DPAPI.
- Give each installation a durable cryptographic identity and show both device fingerprints before trust is accepted.
- Require mutual peer authentication for reconnects, with explicit forget/revoke and re-pair actions.
- Bind every control request, benchmark stream, and RPC tunnel to the authenticated peer and current coordinator lease.
- Add protocol transcript binding, replay protection, monotonic counters, request authorization, bounded timeouts, and rate limits.
- Rekey long-lived sessions and rotate stored credentials after upgrades or suspected compromise.
- Add an authenticated peer allowlist and clear audit events without logging pairing codes, keys, prompts, or personal paths.
- Add tests for replay, downgrade, impersonation, stale pairing codes, unauthorized tunnel creation, revoked peers, corrupted frames, and reconnect recovery.
- Consider a well-reviewed PAKE such as SPAKE2/OPAQUE or certificate-based mutual TLS instead of a custom handshake.

Until this work is implemented, do not expose SharedLocalLLM peer ports outside a trusted private LAN and do not describe the peer layer as production-hardened authentication.

## Use `async-openai` for inference requests

Consider using the unofficial Rust [`async-openai`](https://github.com/64bit/async-openai) client for requests from SharedLocalLLM to the loopback `llama-server` OpenAI-compatible API.

It would provide typed chat-completion requests and responses, configurable base URLs, SSE token streaming, image and tool-call payloads, and custom request/response types for llama.cpp-specific differences.

Recommended responsibility split:

- Use `async-openai` for the in-app chat client and streaming inference requests.
- Retain Axum for hosting SharedLocalLLM's localhost OpenAI-compatible endpoint.
- Retain direct `reqwest` usage for transparent proxying and llama.cpp-specific endpoints such as `/health`, `/metrics`, and `/props`.
- Avoid adopting a higher-level agent framework such as Rig unless the product later needs agent, RAG, or multi-provider abstractions.

There is currently no official OpenAI-maintained Rust SDK, so dependency maintenance and llama.cpp compatibility should be evaluated before adoption.

## Remote CPU offload over RPC

The peer RPC worker currently exposes only GPU-class devices (GPU, iGPU, accelerator); a connected
machine's CPU is used only as a fallback when that machine has no GPU at all. llama.cpp's RPC client
also reports every RPC device as a GPU regardless of its real type, so the coordinator cannot
currently target a remote CPU at all.

Future work to enable remote CPU offload should include:

- Expose the worker's CPU as an additional RPC device even when a GPU is present (or behind a
  per-cluster toggle) instead of only as a no-GPU fallback.
- Let the coordinator identify which RPC device is the remote CPU so a layer share can be assigned
  explicitly rather than the current even split across all RPC devices.
- Decide placement semantics first: remote CPU alongside remote GPU, or remote CPU only as a last
  resort when the model exceeds combined GPU VRAM.
- Keep raw llama.cpp RPC loopback-only; only the trusted-peer tunnel crosses the LAN.
- Benchmark before promoting: offloading to a remote CPU is generally slower than the local GPU and
  should not be presented as a distributed speedup.

Implemented in the Python backend (`python-backend-migration` branch) as an experimental preview:

- `NativeRpcServer` exposes the worker's CPU as a final RPC device behind the `include_cpu` toggle,
  preserving the no-GPU fallback when a GPU is absent.
- The coordinator identifies the remote CPU as the last freshly-registered RPC device (llama.cpp
  reports every RPC device as GPU type), and computes an explicit `tensor_split` that separates remote
  GPU, remote CPU, and local GPU shares instead of the previous even split.
- `ModelLoadConfig.includeRemoteCpu` plus `kind: "cpu"` layer allocations flow through the UI (a
  "remote CPU" toggle and layer input in the manual split panel) and the peer `rpc_tunnel` handshake.
- Placement remains manual: the user chooses how many layers to assign to the remote CPU. Automatic
  allocation and `fit` classification still consider GPU VRAM only.

Still outstanding: physical two-computer validation, and a benchmark that confirms remote-CPU offload
is not presented as a distributed speedup. RPC device enumeration accumulates across loads because
there is no exported unregister/free symbol; the coordinator caches devices by worker endpoint.
delta, which is sufficient for the controlled single-cluster case.

## Dual-engine inference: pinned `llama-server` for MTP models

Current Qwen releases (Qwen3.6/3.8 class) embed NextN/MTP draft heads in their GGUFs. llama.cpp's
server unlocks them with `--spec-type draft-mtp`, giving roughly 2-3x generation throughput on
reasoning-heavy workloads. Our sidecar links llama.cpp as a library through llama-cpp-python, whose
chat-completion path has no speculation loop, so those models currently run correctly but at base
speed. HauhauCS-style FastMTP sidecars require patched llama.cpp builds and stay explicitly out of
scope: every bundled binary must come from the pinned official runtime manifest with origin, size,
and SHA-256 verification.

Design goals:

- Keep llama-cpp-python as the default engine; introduce a second, pinned official `llama-server.exe`
  used only when the selected model advertises MTP tensors (GGUF metadata) or the user opts in
  per model.
- Preserve all v1 security properties: the server binds loopback only, requires the existing
  per-install bearer key (`--api-key` sourced from the store), never touches the LAN directly, and
  its port is excluded from the user-facing API port range checks.
- Route `runtime.chat`, streaming, cancellation, and the OpenAI-compatible proxy through the child
  server's loopback endpoint instead of in-process calls while that engine is active. llama-server's
  `--reasoning-format deepseek` returns `reasoning_content` natively, which can bypass the name-based
  reasoning splitter in server mode.
- Retain two-computer distribution: llama-server accepts `--rpc`, so the worker's existing RPC daemon
  can serve layer splits in server mode exactly as it does today.
- One engine at a time per machine to avoid double VRAM residency; stop/switch flows must tear the
  previous engine down before the next starts.
- Process lifecycle lives in the Python backend beside `ApiServerManager` (spawn, health via
  `/health`, graceful stop, orphan kill on crash), mirroring the existing dev-watcher patterns.

Suggested phases:

1. Manifest and packaging: add the pinned `llama-server` binary to the runtime manifest with
   checksums, install it beside the sidecar, and verify `/health` reachability. No behavior change.
2. Engine selection and lifecycle: detect MTP capability from GGUF metadata at discovery, add the
   per-model engine choice to load options, and implement spawn/stop/restart management.
3. Chat routing parity: non-streaming chat, then SSE streaming mapped onto the existing
   `ChatStreamEvent` channel, plus cancellation by aborting the in-flight upstream request.
4. Benchmarks and placement: run inference benchmarks against whichever engine serves the cluster,
   record the engine in the result, and refresh `docs/architecture.md` and `docs/testing.md`.

Risks to watch: llama-server release cadence vs the pinned manifest (re-pin deliberately, never
auto-float); extra VRAM from draft heads on small GPUs; Windows firewall prompts must not appear
(loopback bind only); K_P/IQ quant labels remain cosmetic concerns only.
