#!/bin/sh
# Install the SharedLocalLLM Python backend on macOS/Linux.
#
# Apple Silicon uses the prebuilt llama-cpp-python Metal wheel (no CUDA build
# flags, no compiler toolchain needed). Windows keeps using
# setup-python-backend.ps1, which builds the CUDA wheel from source.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="$ROOT/backend"
VENV="$BACKEND/.venv"
PYTHON="$VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
  python3 -m venv "$VENV"
fi

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install "llama-cpp-python==0.3.36"
"$PYTHON" -m pip install -e "$BACKEND[dev]"
echo "SharedLocalLLM Python backend is installed with Metal support."
