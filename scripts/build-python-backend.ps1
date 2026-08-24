$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Python = Join-Path $Backend ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    & (Join-Path $PSScriptRoot "setup-python-backend.ps1")
}
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name sharedlocalllm-backend `
    --distpath (Join-Path $Backend "dist") `
    --workpath (Join-Path $Backend "build") `
    --specpath $Backend `
    --collect-all llama_cpp `
    --collect-all uvicorn `
    --collect-all fastapi `
    (Join-Path $Backend "sidecar_entry.py")
Write-Host "Python sidecar built at backend\dist\sharedlocalllm-backend.exe"
