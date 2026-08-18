$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Venv = Join-Path $Backend ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.12 -m venv $Venv
    } else {
        & (Get-Command python -ErrorAction Stop).Source -m venv $Venv
    }
}

& $Python -m pip install --upgrade pip setuptools wheel ninja
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_ARGS = "-DGGML_CUDA=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON"
$env:FORCE_CMAKE = "1"
& $Python -m pip install --upgrade --force-reinstall --no-cache-dir "llama-cpp-python==0.3.34"
& $Python -m pip install -e "${Backend}[dev]"
Write-Host "SharedLocalLLM Python backend is installed with CUDA + RPC support."
