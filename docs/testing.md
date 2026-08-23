# Testing and release gates

No physical-hardware support claim should be made from mocked or loopback tests. Automated gates
prove deterministic behavior and packaging; the physical matrix separately proves actual peer,
driver, network, and GPU behavior.

## Frontend gate

```powershell
pnpm install --frozen-lockfile
pnpm check
```

`pnpm check` performs a Prettier check, strict TypeScript build, ESLint with zero warnings, Vitest
with coverage, production Vite build, and Chromium browser tests. It does not rewrite source files.
Vitest enforces at least 80% statements, branches, functions, and lines across frontend production
code.

For a focused loop:

```powershell
pnpm test
pnpm test:coverage
pnpm e2e
```

The Playwright suite starts a production Vite preview and exercises the browser-demo path with
semantic roles and stable workflow test IDs. Browser-demo tests do not count as proof that a Tauri
command, local runtime, firewall operation, or real peer works.

## Rust gate

```powershell
cargo fmt --manifest-path src-tauri/Cargo.toml --all --check
cargo test --manifest-path src-tauri/Cargo.toml --all-targets
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
```

Unit coverage should include model/shard/projector discovery, memory fitting, split candidates,
recommendation invalidation, runtime-manifest validation, command construction, redaction, and
lifecycle transitions. Integration tests should use fake sidecars and loopback peers for pairing,
protocol rejection, tunnel integrity, cancellation, API proxying, occupied ports, and cleanup.

## Packaging

```powershell
pnpm build
pnpm tauri build --bundles nsis
```

GitHub Actions builds an unsigned NSIS artifact on Windows. Before a public release, also:

1. Pin official `llama.cpp` and CUDA-runtime assets in the runtime manifest.
2. Verify archive size, SHA-256, executable inventory, startup health, and rollback.
3. Configure a protected Windows code-signing certificate outside the repository.
4. Sign the installer and publish its checksum.
5. Install the artifact on clean supported Windows images rather than testing only a dev build.

## Physical peer smoke test (two PCs)

A lighter-weight network/firewall/loopback sanity check to run on one of the two machines before the
manual matrix below. These are smoke checks only, not proof of distributed GPU inference; the
physical two-computer acceptance matrix below remains authoritative for performance and correctness.

### PowerShell harness (`scripts/two-pc-acceptance.ps1`)

```powershell
.\scripts\two-pc-acceptance.ps1 -PeerAddress 10.10.10.2
```

Checks are reported per-check as PASS/FAIL/SKIP with a final `RESULT: PASS|FAIL` and an exit code of
0 on pass or 1 on any FAIL:

1. Firewall rules `SharedLocalLLM Peer Backend` (TCP 49158) and
   `SharedLocalLLM Peer Discovery` (UDP 49157) exist,
   are Enabled, Inbound, Allow, and `Profile = Any`.
2. Something is listening on TCP 49158 with a non-loopback `LocalAddress` (the peer channel binds
   `0.0.0.0`).
3. A UDP endpoint is bound on port 49157.
4. `Test-NetConnection` to `$PeerAddress` port 49158 succeeds.
5. If the packaged Python backend or a llama.cpp sidecar is running, every listener except the
   Python backend's peer port is loopback-only (`127.0.0.1`/`::1`); if none is running the check is
   skipped, not failed.

These are network/firewall/loopback smoke checks, not proof of distributed GPU inference; that
remains the manual matrix below.

## Physical two-computer acceptance

Record exact app/runtime versions, Windows builds, drivers, adapters, models, context, and results.
Test at least two different NVIDIA generations and asymmetric VRAM/RAM without encoding those
specific products into application logic.

- Install and pair without Node.js, Rust, a CUDA toolkit, or LM Studio installed.
- Verify the raw RPC port is unreachable from the peer and another LAN computer.
- Verify the app and API remain loopback-only while both GPUs allocate memory and execute work.
- Run a model that fits one node; compare both valid single-node placements with distributed mode.
- Run a model too large for either GPU but small enough for combined VRAM.
- Run a valid model requiring coordinator RAM spill and confirm the operating-system reserve.
- Enable "remote CPU" offload in the manual split, assign a few layers to the worker's CPU, and
  confirm both GPUs plus the worker CPU allocate work while the model loads. Record that remote-CPU
  offload is presented as an experimental manual placement, not a distributed speedup.
- Exercise text chat from both computers, including SSE and cancellation. Confirm that vision
  attachments return the documented migration limitation instead of silently failing.
- Disconnect during model load and generation; confirm clear failure and complete process cleanup.
- Close and reopen each app independently; confirm the saved peer returns to Reachable without a
  new pairing code. Then use **Nodes > Forget**, confirm both peers require a fresh pairing, and
  verify every configured model path and file remains unchanged.
- Run a distributed benchmark and record the displayed per-computer GPU layer counts. Confirm both
  GPUs allocate memory while `llama-bench` runs and the result is labelled `distributed`; a browser
  demo or command-construction test is not physical acceptance.
- Exercise Ethernet, Wi-Fi, manual IP, static direct Ethernet (`10.10.10.x`), automatic link-local
  (`169.254.x.x`), operation while Windows reports the network as Public (which must now work without
  any profile change), multiple simultaneous adapters (Wi-Fi + Ethernet) with a stable
  manually-selected peer route, and Windows network-category changing while connected (the session
  must not terminate).
- Confirm model directories are unchanged after indexing, benchmarking, chat, and uninstall.

Windows network-category enforcement has been removed: the Public/Private category is informational
only, and the app pairs and launches identically on every category. The automated suite covers
firewall-rule construction (port-scoped, Profile Any) and endpoint parsing for static and
link-local addresses, but Public-profile and multi-adapter behaviour still needs physical two-PC
validation.

Keep failures visible. A benchmark failure record must preserve the actionable error and command
context after secret/path redaction; it must not silently disappear from recommendation results.
