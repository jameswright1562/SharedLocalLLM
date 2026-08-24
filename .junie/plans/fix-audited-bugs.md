---
sessionId: session-260817-203636-v11c
---

# Bug Findings

### Overview

A read-through audit of the SharedLocalLLM codebase (peer channel, inference, capacity/split math, runtime installer, and the front-end mirror) surfaced several concrete defects. They are ordered by severity below with file/line references. The attached `src-tauri/tests/physical_peer.rs` exercises the exact paths affected by the two highest-severity peer bugs (`RpcTunnel` → `bridge`, and worker lifecycle).

### B1 — RPC tunnel `bridge()` is not cancellation-safe (High)

**Where:** `src-tauri/src/peer/tunnel.rs`, `bridge()` lines 81–112 (specifically the `frame_size = left_read.read_u32()` select branch at line 89).

**Problem:** The bidirectional pump uses `tokio::select!` with `left_read.read_u32()` as one branch. `AsyncReadExt::read_u32` / `read_exact` are **not cancellation-safe** — they buffer partially-read bytes in the future's own state. When the other branch (`right_read.read(...)`) becomes ready first, the `read_u32` future is dropped, and any length-prefix bytes it already consumed from the socket are lost. On a busy bidirectional RPC tunnel this desynchronizes the length-prefixed framing and corrupts the stream.

**Impact:** Intermittent RPC tunnel corruption / hangs during distributed inference — exactly the code path a two-computer run relies on. Hard to reproduce, easy to mis-attribute to the network.

### B2 — `StopWorker` does not stop the embedded RPC worker (High)

**Where:** `src-tauri/src/peer/service/worker.rs` `stop_worker` (lines 39–42) and `ensure_worker` (77–124); `src-tauri/src/inference/rpc_worker.rs` `start_rpc_worker` (lines 9–27).

**Problem:** The embedded llama.cpp RPC server is started once at app startup via `ggml_backend_rpc_start_server` on `127.0.0.1:50052`, guarded by a `OnceLock` and with no shutdown path. The peer `StopWorker` handler only sets `state.worker = None` and `state.rpc_target = None`. But `state.worker` is *always* `None` (nothing ever populates it — `ManagedChild` in `worker.rs` is effectively dead code), and `ensure_worker` falls back to the hard-coded `127.0.0.1:50052` regardless of `rpc_target`. So `StopWorker` has essentially **no effect**: the RPC server keeps listening and new tunnels still connect.

**Impact:** The advertised ability to stop the remote compute worker is a no-op; UI/telemetry that assumes the worker stopped is misleading. `ManagedChild` is unreachable dead code.

### B3 — Wrong logits index when sampling (Medium, suspected)

**Where:** `src-tauri/src/inference/generation.rs` line 62: `let next_token = sampler.sample(context, (n_past - 1) as i32);`

**Problem:** `sample(ctx, idx)` expects the **batch logit index** of the token whose logits should be sampled, not the absolute sequence position. During prefill only the final prompt token is decoded with `logits = true`, so its logits live at the last *batch* index of the last chunk (`chunk.len() - 1`); after each single-token decode step the logits are at batch index `0`. Passing the ever-growing absolute position `n_past - 1` reads the wrong (or out-of-range) logits slot rather than the intended last-token logits.

**Impact:** Incorrect / degenerate sampling on prompts longer than the 512-token batch. Needs confirmation against the exact `llama_cpp_4` `sample` semantics, but the index arithmetic is inconsistent with how the batch is built.

### B4 — Front-end split estimate can produce `NaN` and diverges from the Rust source of truth (Medium)

**Where:** `src/services/splitEstimate.ts` line 18 vs `src-tauri/src/commands/cluster/split.rs` lines 101–104.

**Problem:** The two implementations of the same math have drifted:
- Rust: `load_config.context_size.max(4096).min(model.context_length.max(4096))` — the upper bound is clamped to at least 4096.
- TS: `Math.max(4096, Math.min(loadConfig.contextSize, model.contextLength))` — `model.contextLength` is used raw. If it is `undefined`/missing, `Math.min(x, undefined)` is `NaN`, which propagates through `kvMibPerLayer` → `estimatedVramMib` → forces `fits = false`, so `fitLayersByVram` returns an empty split even when a valid one exists.

**Impact:** For models whose context length is not reported, the local (browser/preview) split estimate can silently fail to find any fitting layout. Two parallel implementations also invite future drift.

### B5 — `distribute_layers_by_vram` forces a layer onto zero-VRAM nodes (Low)

**Where:** `src-tauri/src/commands/cluster/split.rs` lines 118–152 (and its TS mirror `distributeLayersByVram`).

**Problem:** `reserved_per_node` unconditionally assigns at least one layer to every node when `total_layers >= node_count`, even for a node reporting `vram_available_gb == 0`. This can generate a split that provably cannot fit on that node yet is still offered as an automatic recommendation.

**Impact:** Automatic allocation can recommend an impossible placement on a GPU-less/insufficient node. Lower severity because manual estimate then reports `fits = false`.

### Notes / lower-confidence observations

- `discover()` in `src-tauri/src/peer/discovery.rs` (line 110) binds UDP `49157` without `SO_REUSEADDR`; it cannot coexist with another listener on the same port on the host. Marginal, but relevant to the `peer_broadcasts_discovery_announcements` physical test if the app is already listening.
- Streaming detokenization in `generation.rs` line 69 uses `String::from_utf8_lossy` per token, which yields replacement characters for multi-byte glyphs split across tokens (known limitation, cosmetic).

# Fix Design

### B1 — Make the tunnel pump cancellation-safe

**File:** `src-tauri/src/peer/tunnel.rs` (`bridge`).

Replace the single-loop `select!` that awaits `read_u32()` directly with one of:
- Split each direction into its own spawned task (`left→right` and `right→left`), each owning its half of the split socket and looping without `select!`, so no in-flight `read_u32`/`read_exact` is ever cancelled. Join both; when either ends, drop the peer to unblock the other.
- Or keep a single loop but only place *cancellation-safe* primitives in `select!` (e.g. an initial `read` into a small header buffer, then a fully-awaited assembly of the 4-byte length outside the select), never a bare `read_u32()`/`read_exact()` branch.

Keep the existing `TUNNEL_CHUNK` frame-size guard and per-direction buffers (each task needs its own buffer to avoid the current shared-`buffer` aliasing).

### B2 — Give the embedded RPC worker a real stop path (or make `StopWorker` honest)

**Files:** `src-tauri/src/inference/rpc_worker.rs`, `src-tauri/src/peer/service/worker.rs`.

Two viable directions (to be confirmed with the user in review):
- **Preferred:** Introduce a stop handle for the embedded server (e.g. a shutdown flag / cancellation the worker thread checks, plus resetting the `OnceLock`-style guard) so `StopWorker` truly stops accepting RPC connections, and have `ensure_worker` (re)start it on demand instead of always falling back to `127.0.0.1:50052`.
- **Minimal:** If the embedded server genuinely cannot be stopped in this preview, remove the dead `ManagedChild` path and change `StopWorker` semantics/telemetry so it no longer claims to stop a worker it cannot stop (report "worker remains resident" rather than success).

### B3 — Sample from the correct batch logit index

**File:** `src-tauri/src/inference/generation.rs`.

Track the batch index of the token decoded with `logits = true` and pass that to `sampler.sample`, rather than the absolute position `n_past - 1`:
- After prefill, the sample index is the last chunk's `chunk.len() - 1`.
- After each single-token decode, it is `0` (the only token in that batch).

Validate against the `llama_cpp_4` `sample` signature and existing engine tests before changing.

### B4 — Align the front-end estimate with Rust and guard against missing context length

**File:** `src/services/splitEstimate.ts`.

- Clamp the upper bound the same way Rust does: `Math.max(4096, Math.min(loadConfig.contextSize, Math.max(4096, model.contextLength ?? 0)))`, eliminating the `NaN` path.
- Add a unit test covering a model with `contextLength` undefined/0 to lock the behaviour and prevent future drift from `capacity.rs`.

### B5 — Do not reserve layers on nodes that cannot hold them

**Files:** `src-tauri/src/commands/cluster/split.rs` and `src/services/splitEstimate.ts`.

Compute `reserved_per_node` only across nodes with positive available VRAM (or skip zero-VRAM nodes entirely from the automatic distribution), keeping the existing "assign at least one layer" behaviour for capable nodes only. Mirror the change in the TS implementation and add a regression test with a zero-VRAM node.

### Cross-cutting

- Keep all peer sockets loopback-only where required; do not change bind addresses.
- Preserve existing error-payload codes/messages and redaction guarantees.
- Keep each touched file single-purpose and under the ~300-line guideline.

# Testing

### Validation approach

Each fix is paired with an automated check that can run without the physical two-computer rig, plus a note on the physical acceptance path where relevant.

### B1 — Tunnel cancellation safety

- Add a loopback unit/integration test that runs `bridge` between two in-process `TcpStream`s and drives **simultaneous bidirectional** traffic (interleaved length-prefixed frames both ways), asserting every byte arrives intact and in order over many iterations.
- Add a stress variant with small random frame sizes to surface partial-read cancellation regressions.

### B2 — Worker stop semantics

- Add a test that issues `StopWorker` and then asserts the intended post-condition of the chosen design (either: a subsequent RPC connect fails until restart, or: the response/telemetry accurately reflects that the resident worker was not stopped). Remove/adjust any assertion that currently implies a no-op stop "succeeded".

### B3 — Sampling index

- Add a deterministic generation test (temperature 0 / greedy) on a small fixture model asserting stable, sensible output for a prompt longer than the 512-token batch, guarding against the wrong-index regression. Rely on existing engine test scaffolding if present.

### B4 — Front-end estimate

- Extend `src/services/appService.test.ts` / add a `splitEstimate` test asserting no `NaN` in `estimatedVramMib` and correct `fits` when `model.contextLength` is undefined/0, and that results match the Rust clamp for representative inputs.

### B5 — Zero-VRAM distribution

- Add a `distribute_layers_by_vram` unit test (Rust and TS) with one zero-VRAM node asserting it receives 0 layers and that all layers are still preserved across capable nodes.

### Regression gate

- Run existing suites: `cargo test --manifest-path src-tauri/Cargo.toml --all-targets`, `cargo clippy ... -D warnings`, `cargo fmt --check`, and `pnpm test` / `pnpm typecheck` / `pnpm lint`, keeping the frontend coverage gate satisfied.
- The physical two-computer acceptance in `docs/testing.md` remains the ultimate check for B1/B2 and is explicitly out of scope for automated CI.

# Delivery Steps

### ✓ Step 1: Fix the RPC tunnel cancellation-safety bug (B1)
`bridge()` in `peer/tunnel.rs` moves bytes in both directions without ever cancelling an in-flight framed read, eliminating tunnel corruption.

- Refactor `bridge` in `src-tauri/src/peer/tunnel.rs` to avoid a bare `read_u32()`/`read_exact()` inside `tokio::select!` (e.g. one spawned task per direction, each owning its socket half and its own buffer).
- Preserve the `TUNNEL_CHUNK` frame-size guard and loopback behaviour; ensure clean teardown when either side closes.
- Add a loopback integration test driving simultaneous bidirectional length-prefixed traffic and asserting byte-exact, in-order delivery over many iterations.

### * Step 2: Make StopWorker honest about the embedded RPC worker (B2)
`StopWorker` either truly stops the embedded llama.cpp RPC server or accurately reports that it cannot, and dead code is removed.

- Decide (in review) between a real shutdown path for `ggml_backend_rpc_start_server` in `src-tauri/src/inference/rpc_worker.rs` vs. honest no-op semantics.
- Update `stop_worker`/`ensure_worker` in `src-tauri/src/peer/service/worker.rs` accordingly and remove the unreachable `ManagedChild` path if it stays dead.
- Add a test asserting the chosen post-`StopWorker` behaviour and adjust any telemetry/messages that currently imply a successful stop.

###   Step 3: Correct the sampler logits index in generation (B3)
`generate()` samples from the batch logit index of the last decoded token instead of the absolute sequence position.

- In `src-tauri/src/inference/generation.rs`, track and pass the correct batch index to `sampler.sample` (last chunk's `chunk.len()-1` after prefill, `0` after each decode step).
- Verify against the `llama_cpp_4` `sample` signature before changing.
- Add a greedy/deterministic generation test on a fixture model with a prompt longer than the 512-token batch to lock correct output.

###   Step 4: Align the front-end split estimate and fix the NaN path (B4)
`splitEstimate.ts` produces the same clamped context size as `capacity.rs`/`split.rs` and never yields `NaN` for models missing a context length.

- Update the context clamp in `src/services/splitEstimate.ts` to mirror the Rust `context_size.max(4096).min(model.context_length.max(4096))` logic and guard `model.contextLength` being undefined/0.
- Add a test asserting no `NaN` in `estimatedVramMib`/`fits` for a model with missing context length and parity with the Rust output for representative inputs.

###   Step 5: Stop reserving layers on zero-VRAM nodes (B5)
Automatic layer distribution no longer forces a layer onto a node that has no available VRAM, in both Rust and TS implementations.

- Adjust `distribute_layers_by_vram` in `src-tauri/src/commands/cluster/split.rs` so `reserved_per_node` applies only to nodes with positive available VRAM (or excludes zero-VRAM nodes), while still preserving all layers across capable nodes.
- Mirror the change in `distributeLayersByVram` in `src/services/splitEstimate.ts`.
- Add Rust and TS regression tests with a zero-VRAM node asserting it gets 0 layers and total layers are preserved.