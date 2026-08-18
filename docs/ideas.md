# SharedLocalLLM ideas

This file records worthwhile follow-up work intentionally outside the current v1 delivery gate.

## Improved peer authentication and authorization

The v1 peer channel is intended only for two computers controlled by the same person on a trusted Windows Private network. It uses an expiring six-digit pairing code with an encrypted Noise channel, rejects incorrect codes, and keeps raw `llama.cpp` RPC plus the model API on loopback. This is adequate for the current local preview, but it is not the desired long-term trust model.

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

## Remember last used cluster settings

When starting a cluster with a model, the program should remember the user's last used settings (model selection, quantization, GPU/CUDA settings, layer mappings, etc.) and pre-populate them for the next cluster launch. This would improve workflow efficiency by reducing repetitive manual configuration.

Implementation considerations:
- Store settings per-model or globally as a JSON configuration
- Allow user to clear/reset remembered settings
- Consider privacy implications of persisted inferencing preferences
