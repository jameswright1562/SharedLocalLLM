$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "backend\.venv\Scripts\python.exe"
if (Test-Path $Python) {
    & $Python -c "import fastapi, llama_cpp, psutil, uvicorn" 2>$null
    if ($LASTEXITCODE -eq 0) { exit 0 }
}
& (Join-Path $PSScriptRoot "setup-python-backend.ps1")
