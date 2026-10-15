#!/bin/sh
# macOS/Linux placeholder for the pinned llama-server install.
#
# No macOS llama-server binary is pinned yet (the manifest only pins the
# Windows CUDA assets), so there is nothing to download. The built-in
# llama-cpp-python engine (Metal on Apple Silicon) covers inference,
# benchmarks, and RPC workers; the app falls back to it automatically.
echo "No pinned llama-server binary for this platform yet; using the built-in Metal engine."
