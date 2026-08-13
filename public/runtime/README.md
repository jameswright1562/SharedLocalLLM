# Runtime manifest

`llama-cpp-manifest.json` is metadata only. Its release is pinned to official upstream asset URLs,
sizes, and SHA-256 values; it does not contain the large binaries.

For future updates, use the following release shape:

```json
{
  "tag": "bNNNN",
  "publishedAt": "YYYY-MM-DDTHH:mm:ssZ",
  "llamaCpp": {
    "url": "https://github.com/ggml-org/llama.cpp/releases/download/bNNNN/llama-bNNNN-bin-win-cuda-12.4-x64.zip",
    "size": 123456789,
    "sha256": "64 lowercase hexadecimal characters"
  },
  "cudaRuntime": {
    "url": "https://github.com/ggml-org/llama.cpp/releases/download/bNNNN/cudart-llama-bin-win-cuda-12.4-x64.zip",
    "size": 123456789,
    "sha256": "64 lowercase hexadecimal characters"
  }
}
```

The application must reject a disabled manifest, a missing asset, any non-HTTPS GitHub URL,
an unexpected archive entry, a size mismatch, or a digest mismatch. Retain the preceding
verified runtime until the newly installed runtime passes its executable health checks.

The required RPC executable is `ggml-rpc-server.exe`. The application must never download or
trust an archive merely because its filename matches the pattern.
