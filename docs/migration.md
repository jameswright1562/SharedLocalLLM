# Migration: llama-cpp-4 API and peer/RPC refactor

This document records the migration applied on the `use-new-crate` branch that replaced the old
`llama.cpp` wrapper usage with `llama-cpp-4 0.5.1` (+ direct `llama-cpp-sys-4`), and the peer/RPC
refactor it surfaced ("Remove pairing", commit `d06d0b8`). It is a migration record, not a
regression plan: the automated gates described at the end are the acceptance criteria.

## Background

`llama-cpp-4 0.5.1`'s high-level `rpc` module wraps the **old** ggml RPC API, while its sibling
`llama-cpp-sys-4 0.5.1` vendors a llama.cpp that ships the **rewritten** RPC API
(`ggml/src/ggml-rpc/`). Enabling `rpc` through `llama-cpp-4` therefore fails to compile against
the generated sys bindings.

SharedLocalLLM never uses `llama_cpp_4::rpc`; it calls `llama_cpp_sys_4` directly. The fix is to
stop forwarding the `rpc` feature through the high-level crate and enable it on the sys crate.

## Dependency changes

`src-tauri/Cargo.toml`:

- `llama-cpp-4 = { version = "0.5.1", features = ["cuda", "openmp", "mtmd"] }` (the `rpc` feature
  was removed here).
- Added `llama-cpp-sys-4 = { version = "0.5.1", features = ["rpc"] }`.

Cargo feature unification keeps a single `llama-cpp-sys-4 0.5.1` with `rpc` + `cuda`/`openmp`/`mtmd`
(the latter three forwarded through `llama-cpp-4`). Both crates only publish version `0.5.1`
(crates.io, checked at migration time). Upstream repository: `https://github.com/eugenehp/llama-cpp-rs`.

## Imports and the `Result` shadowing fix

`llama_cpp_4::prelude::*` re-exports the crate's own `Result<T>` alias (error type
`LLamaCppError`), which shadows `std::result::Result` in modules that also use
`Result<_, crate::types::ErrorPayload>`. That caused `E0107` (type alias arity), `E0277`
(`?` conversion to `LLamaCppError`), and `E0308` errors.

`src/inference/engine.rs` and `src/inference/generation.rs` now import the concrete types instead of
the prelude glob:

```rust
use llama_cpp_4::{
    model::LlamaChatMessage, AddBos, LlamaBackend, LlamaBatch, LlamaContext, LlamaModel,
    LlamaSampler, Special,
};
```

Note that `LlamaChatMessage` is only exported at `llama_cpp_4::model::LlamaChatMessage`; all other
core types are re-exported at the crate root.

## API mapping (old call -> llama-cpp-4)

| Old call (pre-refactor)                  | llama-cpp-4 0.5.1 replacement                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `model.chat_template()`                  | `model.apply_chat_template(None, &[LlamaChatMessage], true)` -> `Result<String, ApplyChatTemplateError>`      |
| `model.tokenize(&prompt, false)`         | `model.str_to_token(&prompt, AddBos::Always)` -> `Result<Vec<LlamaToken>, StringToTokenError>`                |
| `model.sampler_default().temperature(t)` | `LlamaSampler::chain_simple([LlamaSampler::temp(t as f32), LlamaSampler::dist(0)])`                           |
| `context.clear()`                        | removed; fresh context per model load. If a KV reset is ever needed, `LlamaContext::clear_kv_cache_seq(...)`  |
| `context.decode(backend, batch)`         | `context.decode(&mut batch)` -> `Result<(), DecodeError>` (backend no longer required)                        |
| `context.sample(&sampler)`               | `sampler.sample(&ctx, idx)` -> `LlamaToken` (sampler is owned separately, no error result)                    |
| `model.token_to_piece(token)`            | `model.token_to_bytes(token, Special::Plaintext)` -> `Result<Vec<u8>>`, render with `String::from_utf8_lossy` |
| `model.token_eos()`                      | unchanged (`model.token_eos() -> LlamaToken`)                                                                 |
| `batch.add(token, pos, &[0], logits)`    | same signature, but returns `Result<(), BatchAddError>`; `LlamaToken` derives `PartialEq`/`Eq`/`Copy`         |

Prompt construction:

- Chat models use `apply_chat_template` with `LlamaChatMessage::new(role, content)` (returns
  `Result<_, NewLlamaChatMessageError>`); the system prompt is prepended as a `"system"` message.
- Models without a chat template fall back to the plain `build_prompt_fallback`.

Position tracking changed: the prefill loop now uses a running `n_past` offset (the old code used
the per-chunk index `i` as the position, which is wrong for prompts longer than one batch), and the
last prompt token requests logits (`logits = position + 1 == tokens.len()`). Sampling uses
`idx = n_past - 1` after decode.

## Direct sys usage (unchanged behaviour, new signatures)

`src/inference/rpc.rs` and `src/inference/rpc_worker.rs` call `llama_cpp_sys_4` directly and must
match the new RPC API:

- `ggml_backend_rpc_add_server(endpoint: *const c_char) -> ggml_backend_reg_t`
- `ggml_backend_rpc_start_server(endpoint: *const i8, cache_dir: *const i8, n_threads: usize,
n_devices: usize, devices: *mut *mut ggml_backend_device)`

## Peer/RPC refactor residue fixed

`PeerServerConfig` (from "Remove pairing") now has exactly:

`bind, device_id, device_name, capabilities, rpc_target: SocketAddr, catalogue, api_key, api_port`.

Removed fields: `pairing_code`, `trusted_peers`, `rpc_override`, `rpc_binary`.

- `src/peer/service/connection.rs` seeds `ServerState.rpc_target` from `config.rpc_target`.
- `src/peer/service/worker.rs` `ensure_worker` returns the configured loopback target after a
  connectivity probe, falling back to `127.0.0.1:50052`. The embedded RPC worker is started at
  application startup (`start_rpc_worker` in `src/lib.rs`) and replaces the external
  `ggml-rpc-server.exe` process. `connect_rpc` still enforces loopback-only targets.
- `src/commands/pairing/lifecycle.rs` supplies `catalogue`/`api_key`/`api_port` and a loopback
  `rpc_target` of `127.0.0.1:50052`.
- `tests/peer_transport.rs` uses `rpc_target` instead of the removed fields, and its echo server
  now accepts multiple connections because `ensure_worker` performs a probe connection to the RPC
  target before the real tunnel stream.

## Dead code removed

`src/commands/cluster/control.rs` dropped the health-check helpers that belonged to the old
external `llama-server` era and were never called after the embedded inference engine:
`wait_for_health`, `server_exited`, `health_error`, `runtime_log_tail`, and the
`HEALTH_TIMEOUT`/`HEALTH_POLL_INTERVAL`/`HEALTH_REQUEST_TIMEOUT` constants.

## Open items

- **RPC frame endianness.** `src/peer/tunnel.rs` frames the client<->peer leg with tokio's
  big-endian `read_u32`/`write_u32`. llama.cpp's `rpc_frame_send` writes the size prefix in native
  byte order (little-endian on x86). The app-internal path is self-consistent, but a real llama.cpp
  RPC client talking to the forwarder may disagree on byte order. Verify during the physical
  two-computer acceptance in `docs/testing.md`; automated loopback tests do not cover this.
- **KV cache persistence.** `context.clear()` was removed. The context is created fresh per model
  load, so a single `Generate` sees a clean cache, but repeated `Generate` calls on the same loaded
  model reuse the KV cache without a reset.

## Verification

Run from `src-tauri`:

- `cargo check --all-targets` — passes, zero warnings.
- `cargo test --all-targets` — 34 tests, 0 failures (3 physical tests in `tests/physical_peer.rs`
  remain ignored; they require the real two-computer setup).
- `cargo clippy --all-targets -- -D warnings` — passes.
- `cargo fmt --all --check` — passes.
