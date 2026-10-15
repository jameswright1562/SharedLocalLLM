#!/bin/sh
# Ensure the backend virtualenv is importable, installing it when needed.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/backend/.venv/bin/python"
if [ -x "$PYTHON" ] && "$PYTHON" -c "import fastapi, llama_cpp, psutil, uvicorn, sharedlocalllm_backend" 2>/dev/null; then
  exit 0
fi
exec sh "$(dirname "$0")/setup-python-backend.sh"
