# Architecture

SharedLocalLLM uses one Tauri application on both Windows computers. Roles are selected at runtime,
not encoded in separate builds.

## Components

- **React renderer:** setup, nodes, models, network measurements, recommendations, chat, API setup,
  and diagnostics. It receives typed events but cannot start arbitrary executables.
- **Rust application core:** trusted identities, peer protocol, capability discovery, catalogue,
  profile selection, process supervision, API proxy, and persistence.
- **Runtime manager:** installs only a pinned official `llama.cpp` release whose archive size and
  SHA-256 match the bundled manifest. A previous verified runtime remains available for rollback.
- **Coordinator runtime:** `llama-server.exe` owns the selected model and connects to the local and
  remote GPU devices using layer split.
- **Worker runtime:** `ggml-rpc-server.exe` listens on an ephemeral loopback port. It is reachable
  only through the Rust process's authenticated peer tunnel.
- **Local API:** binds to `127.0.0.1`. Chat from the worker is proxied over the authenticated peer
  channel to the computer that launched `llama-server`.

## Session lifecycle

```text
unpaired -> paired -> measuring -> ready -> loading -> running -> draining -> stopped
                                      |          |
                                      +-> failed <-+
```

One renewable coordinator lease prevents both peers from loading competing sessions. A role change
first stops accepting work, drains active requests within a deadline, tears down managed processes,
and then transfers the lease. A disconnected peer makes distributed generation fail with a clear
error; single-node retry is offered only if the selected model fits.

Child executables run in Windows Job Objects. Stopping the cluster or losing the supervising app
terminates the process tree. Normal window close hides to the tray while a session is active.

## Peer protocol and trust

Discovery uses a small UDP announcement on private interfaces. Manual private IPv4 entry reaches
the same pairing flow. One computer shows a six-digit pairing code; the other enters it. The code
host also advances once the incoming pair is persisted. Confirmation stores each peer identity and
channel key through Windows DPAPI.

All later peer traffic uses a Noise-encrypted, versioned application handshake. One private-network
TCP listener multiplexes control messages, the bounded RPC byte tunnel, network tests, catalogue
metadata, worker stop, and proxied chat. Incompatible protocol versions fail before launch with
upgrade guidance. Stronger production-grade authentication remains deferred.

Security invariants:

- Raw RPC binds only to `127.0.0.1` and is never advertised, firewall-opened, or routed to the LAN.
- Local API binds only to `127.0.0.1`; it requires a per-install bearer key.
- Public Windows network profiles cannot launch or accept a cluster.
- Peer certificates, pairing material, and API keys are DPAPI-protected, not stored in SQLite.
- Logs redact secrets, prompts, image content, and personal path prefixes.
- `llama-server` filesystem, shell, MCP, and agent tools stay disabled.

These controls reduce exposure; they do not turn upstream experimental RPC into a safe service for
untrusted networks. A direct RPC socket must be treated as a security defect.

## Model and inference data flow

Each node indexes its own read-only sources. Peer catalogue metadata (names and locations) can be
merged for display, but model bytes stay in place and launch requires a local GGUF. Split GGUF
shards become one record; adjacent `mmproj` files are associated with eligible vision models. The
computer that clicks Launch is the coordinator.

The recommendation engine performs these stages:

1. Reserve operating-system memory and a per-GPU safety margin.
2. Determine whether each single node, combined VRAM, or combined GPU plus coordinator RAM can fit
   the requested context.
3. Build a layer-split seed proportional to usable VRAM.
4. Benchmark valid single-node, seed, and nearby distributed profiles.
5. Recommend the fastest valid result and retain the measurements with their complete cache key.

`llama-bench` records warmup, three `pp512` and `tg128` repetitions, load time, peak resources, and
the full redacted error on failure. Recommendations never assume a particular GPU model or promise
a token rate before measurement.

## State and compatibility

JSON settings store non-secret preferences, peer records, and benchmark history. Chat stays in
renderer memory for the session. Credentials are stored separately with DPAPI. Model directories
remain untouched.

Cache keys include model fingerprint, both hardware identities, drivers, runtime version, requested
context, and route/adapter identity. Any change invalidates the result. Peer messages and persisted
records are schema-versioned; unsupported future versions are rejected rather than guessed.

The runtime manifest in `public/runtime` is part of the release trust root. Release engineering must
pin exact upstream assets and hashes, verify expected archive entries, and exercise new executables
before changing the active release.
