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
- **Local API:** binds to `127.0.0.1`; a worker-side request is forwarded through the tunnel to the
  active coordinator.

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
the same pairing flow. Pairing performs an ephemeral key exchange and displays a derived six-digit
comparison code on both computers. Confirmation persists each peer identity through Windows DPAPI.

All later peer traffic uses mutual TLS and a versioned application handshake. One private-network
TCP listener multiplexes control messages, the bounded RPC byte tunnel, network tests, and proxied
API requests. Incompatible protocol/runtime versions fail before launch with upgrade guidance.

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

Each node indexes its own read-only sources. The merged catalogue carries the file owner and a
stable fingerprint, but not model bytes. Split GGUF shards become one record; adjacent `mmproj`
files are associated with eligible vision models. The node holding the selected file coordinates
unless both hold it and measured performance selects the other node.

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

SQLite stores non-secret settings, per-node model metadata, benchmark history, conversations, and
launch profiles. Credentials are stored separately with DPAPI. Model directories remain untouched.

Cache keys include model fingerprint, both hardware identities, drivers, runtime version, requested
context, and route/adapter identity. Any change invalidates the result. Peer messages and persisted
records are schema-versioned; unsupported future versions are rejected rather than guessed.

The runtime manifest in `public/runtime` is part of the release trust root. Release engineering must
pin exact upstream assets and hashes, verify expected archive entries, and exercise new executables
before changing the active release.
