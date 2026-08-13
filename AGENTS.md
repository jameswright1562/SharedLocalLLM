# SharedLocalLLM agent coordination

Last updated by root integration: 2026-08-13 (active implementation)

## Mission and current task

Build the greenfield SharedLocalLLM Windows application end to end: pair exactly two trusted
computers, discover local GGUF models, measure the link and hardware, recommend a valid placement,
and run one model across local and remote compute through a protected `llama.cpp` RPC topology.

The current task is implementation and integration of the approved plan. This repository is not yet
physically validated on a two-computer GPU setup; keep that distinction explicit in code, tests, and
documentation.

## Non-negotiable product constraints

- Do not hardcode or optimize for the user's example GPU/RAM specifications. Detect capabilities.
- Support optional LM Studio discovery and arbitrary user-selected read-only model directories.
- Keep model files in place. Never move, rename, overwrite, or delete them.
- Raw `llama.cpp` RPC must bind to loopback only and must never be exposed to the LAN.
- The inference API must bind to loopback only and require a per-install bearer key.
- Peer traffic must use the authenticated application tunnel; refuse cluster launch on a Windows
  Public network profile.
- Treat upstream RPC and multimodal support as experimental. Do not promise distributed speedups or
  untested physical-hardware compatibility.
- Preserve actionable errors after redacting secrets, prompts, image content, and personal paths.
- Do not enable `llama.cpp` filesystem, shell, MCP, or agent tools.
- Downloads must use a pinned official runtime manifest and pass origin, size, SHA-256, archive-entry,
  and executable-health checks before activation.

## Ownership map

| Workstream       | Owner          | Exclusive areas / responsibility                                                                           |
| ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------- |
| Rust core        | `rust_core`    | `src-tauri/**`; native commands, domain logic, runtime/process/network/security foundations and Rust tests |
| Desktop UI       | `desktop_ui`   | `src/**`; React application, browser service/demo, accessibility and frontend unit tests                   |
| Tooling and docs | `tooling_docs` | Root tooling/config, `tests/**`, `.github/**`, `scripts/**`, `docs/**`, `public/**`, repository metadata   |
| Integration      | `root`         | Cross-workstream decisions, final review, progress updates, full validation and user handoff               |

Agents share one working directory. Files from another owner's area may change at any time; do not
revert or rewrite them. Send the owner or root a precise finding instead.

## Live progress

Only the root integrator updates this table after this initial seed. This single-writer rule avoids
conflicts and stale competing status reports.

| Workstream         | State    | Current evidence / next action                                                                                                             |
| ------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `rust_core`        | Complete | Native implementation is green with 24/24 tests, formatting, Clippy warnings-as-errors, Cargo build, and the version 0.1.2 release bundle. |
| `desktop_ui`       | Complete | Frontend is green with 39/39 tests, coverage above 86% in every dimension, production build, and 4/4 browser workflows.                    |
| `tooling_docs`     | Complete | Junior-friendly README, developer docs, pinned runtime manifest, CI, release packaging, and checksum workflow are in place.                |
| `root` integration | Active   | Version 0.1.2 full gates and packaging are green; install it on both computers and complete physical pairing/model acceptance.             |

## Validation checklist

Update results with exact commands and outcomes; do not mark partial or simulated checks as physical
acceptance.

- [x] `pnpm install` — dependency graph and lockfile created successfully.
- [x] `pnpm peers check` — no peer dependency issues.
- [x] `pnpm format:check` — all matched repository files use Prettier formatting.
- [x] `pnpm typecheck` — strict TypeScript project build passes.
- [x] `pnpm lint` — ESLint passes with zero warnings.
- [x] `pnpm test` — 39/39 frontend tests pass.
- [x] `pnpm test:coverage` — 39/39 tests passed with 92.04% statements, 86.52% branches,
      92.03% functions, and 93.54% lines.
- [x] `pnpm build` — production Vite build succeeds.
- [x] `pnpm e2e` — four of four Chromium browser-demo workflows pass.
- [x] `cargo fmt --manifest-path src-tauri/Cargo.toml --all --check` — passes.
- [x] `cargo test --manifest-path src-tauri/Cargo.toml --all-targets` — 24/24 tests pass.
- [x] `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings` — passes.
- [x] `cargo build --manifest-path src-tauri/Cargo.toml` — development build succeeds.
- [x] Production readability audit — every production file changed for version 0.1.2 is under 300
      lines; the largest changed file is `src-tauri/src/commands/pairing.rs` at 284 lines. The
      pre-existing `src/components/SetupStepContent.tsx` remains 306 lines.
- [x] `pnpm tauri build --bundles nsis` — version 0.1.2 built in the isolated
      `src-tauri/target-package` target while preserving the running development app; SHA-256
      `28444fe5fd7822aa92114a8080c22fff9959246841b24ce587eaae387d2f9abd`.
- [ ] Physical two-computer acceptance in `docs/testing.md`; automated loopback/demo tests do not
      satisfy this item.

## Working rules

- Preserve other agents' and the user's work. Never use destructive Git commands or rewrite an
  unrelated dirty file.
- Do not initialize Git, commit, push, or publish unless the user explicitly requests it. This task
  currently authorizes none of those actions.
- Use tests first for behavior changes and retain the 80% global frontend coverage gate.
- Keep source files single-purpose and ideally under 300 lines. Split large UI or native modules by
  coherent responsibility instead of compressing code or weakening readability.
- Prefer typed, narrow Tauri commands and versioned peer/persistence interfaces. Treat renderer and
  peer input as untrusted.
- Keep checks non-mutating. Formatting may be applied deliberately by the owning agent, but CI and
  `pnpm check` use check-only commands.
- Record the real error when a benchmark, runtime action, or validation command fails. Separate an
  unrelated environment failure from a product failure.
- Root is the sole updater of the **Live progress**, **Validation checklist**, and **Current blockers**
  sections after this file is created. Other agents report status to root instead of editing them.

## Current blockers

- No remaining automated code or packaging blocker is known.
- Physical two-computer GPU acceptance remains pending and must follow `docs/testing.md`; loopback and
  browser-demo tests are not a substitute for that hardware check.
- Stronger production-grade peer authentication is explicitly deferred to `ideas.md`; v1 remains a
  trusted-private-LAN preview.
