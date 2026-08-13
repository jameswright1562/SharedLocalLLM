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
- Exercise text and supported vision chat from both computers, including SSE and cancellation.
- Disconnect during model load and generation; confirm clear failure and complete process cleanup.
- Close and reopen each app independently; confirm the saved peer returns to Reachable without a
  new pairing code. Then use **Nodes > Forget**, confirm both peers require a fresh pairing, and
  verify every configured model path and file remains unchanged.
- Run a distributed benchmark and record the displayed per-computer GPU layer counts. Confirm both
  GPUs allocate memory while `llama-bench` runs and the result is labelled `distributed`; a browser
  demo or command-construction test is not physical acceptance.
- Exercise Ethernet, Wi-Fi, manual IP, Private/Public profile behavior, firewall repair, occupied API
  port, runtime mismatch, failed runtime update, and rollback.
- Confirm model directories are unchanged after indexing, benchmarking, chat, and uninstall.

Keep failures visible. A benchmark failure record must preserve the actionable error and command
context after secret/path redaction; it must not silently disappear from recommendation results.
