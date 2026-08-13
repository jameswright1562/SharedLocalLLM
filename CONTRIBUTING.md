# Contributing

Keep changes scoped, readable, and safe for a user running large local models on two personal
computers. Do not commit models, downloaded runtimes, generated installers, credentials, diagnostic
bundles, or personal paths.

## Workflow

1. Create or update a failing test for the behavior.
2. Make the smallest implementation change that passes it.
3. Run `pnpm check` and the Rust formatting, test, and Clippy commands in
   [docs/testing.md](docs/testing.md).
4. Update user or architecture documentation when behavior, security boundaries, ports, or protocol
   shapes change.

Use accessible names and semantic controls in the UI. Keep Tauri commands narrowly typed; renderer
input is untrusted. Preserve model directories as read-only. Never expose raw RPC or the inference
API to the LAN, weaken the Private-network check, log prompts/secrets, or enable upstream filesystem,
shell, MCP, or agent tools.

Pull requests should state the user-visible outcome, tests run with exact results, and any untested
hardware assumption. Runtime updates must pin official URLs, byte sizes, and SHA-256 values and
document verification evidence. Do not insert placeholder hashes or silently follow a `latest` URL.
