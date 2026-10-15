#!/bin/sh
# Package the Python sidecar with PyInstaller (macOS/Linux).
#
# Produces backend/dist/sharedlocalllm-backend (extension-less), which the
# tauri.conf resources glob ships as backend/sharedlocalllm-backend.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="$ROOT/backend"
PYTHON="$BACKEND/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  sh "$(dirname "$0")/setup-python-backend.sh"
fi
"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name sharedlocalllm-backend \
  --distpath "$BACKEND/dist" \
  --workpath "$BACKEND/build" \
  --specpath "$BACKEND" \
  --collect-all llama_cpp \
  --collect-all uvicorn \
  --collect-all fastapi \
  "$BACKEND/sidecar_entry.py"
echo "Python sidecar built at backend/dist/sharedlocalllm-backend"
