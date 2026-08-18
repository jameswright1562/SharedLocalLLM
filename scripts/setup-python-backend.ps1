$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Venv = Join-Path $Backend ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

function Import-MsvcEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw @"
Visual Studio Build Tools were not found.

Install Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload, then rerun:
  pnpm backend:install
"@
    }

    $installPath = & $vswhere `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $installPath) {
        throw @"
No Visual Studio installation with C++ tools was found.

Install Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload, then rerun:
  pnpm backend:install
"@
    }

    $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        throw "Expected vcvars64.bat at '$vcvars' but it was not found."
    }

    cmd /c "`"$vcvars`" && set" | ForEach-Object {
        if ($_ -match '^(?<key>[^=]+)=(?<val>.*)$') {
            Set-Item -Path "env:$($Matches.key)" -Value $Matches.val
        }
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Failed to initialize the MSVC compiler environment from '$vcvars'."
    }
}

function Ensure-BuildToolOnPath {
    param([string]$ToolName, [string[]]$CandidatePaths)

    if (Get-Command $ToolName -ErrorAction SilentlyContinue) {
        return
    }

    foreach ($candidate in $CandidatePaths) {
        if (Test-Path $candidate) {
            $directory = Split-Path -Parent $candidate
            $env:PATH = "$directory;$env:PATH"
            return
        }
    }

    throw "Required build tool '$ToolName' was not found on PATH."
}

if (-not (Test-Path $Python)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.12 -m venv $Venv
    } else {
        & (Get-Command python -ErrorAction Stop).Source -m venv $Venv
    }
}

Import-MsvcEnvironment
Ensure-BuildToolOnPath -ToolName "cmake.exe" -CandidatePaths @(
    (Join-Path ${env:ProgramFiles} "CMake\bin\cmake.exe")
)
Ensure-BuildToolOnPath -ToolName "ninja.exe" -CandidatePaths @(
    (Join-Path $env:ProgramData "miniconda3\Scripts\ninja.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\Scripts\ninja.exe")
)

& $Python -m pip install --upgrade pip setuptools wheel ninja
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_ARGS = "-DGGML_CUDA=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON"
$env:FORCE_CMAKE = "1"
& $Python -m pip install --upgrade --force-reinstall --no-cache-dir "llama-cpp-python==0.3.34"
& $Python -m pip install -e "${Backend}[dev]"
Write-Host "SharedLocalLLM Python backend is installed with CUDA + RPC support."
