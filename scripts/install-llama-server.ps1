# Installs the pinned llama.cpp server binaries for the dual-engine plan.
# Every asset must come from the pinned manifest: official HTTPS origin,
# exact byte size, exact SHA-256, whitelisted archive entries, and a final
# `llama-server.exe --version` executable health check before activation.
param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Destination) {
    $Destination = Join-Path $Root "backend\runtime\llama-bin"
}
$ManifestPath = Join-Path $Root "public\runtime\llama-cpp-manifest.json"

if (-not (Test-Path $ManifestPath)) {
    throw "Runtime manifest not found at '$ManifestPath'."
}
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
if (-not $Manifest.enabled) {
    throw "The runtime manifest is disabled; refusing to install."
}
if ($Manifest.channel -ne "pinned") {
    throw "Only the 'pinned' channel is accepted; refusing to float to latest."
}

$AllowedPrefix = "https://github.com/ggml-org/llama.cpp/releases/download/"
$RequiredExecutables = @($Manifest.requiredExecutables)
$CacheDir = Join-Path $env:TEMP "sharedlocalllm-llama-server-cache"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

function Get-Sha256Hex {
    param([string]$Path)
    $Algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Stream = [System.IO.File]::OpenRead($Path)
        try {
            return ([System.BitConverter]::ToString($Algorithm.ComputeHash($Stream))).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $Stream.Dispose()
        }
    }
    finally {
        $Algorithm.Dispose()
    }
}

try {
    foreach ($AssetKey in @("llamaCpp", "cudaRuntime")) {
        $Asset = $Manifest.release.$AssetKey
        if (-not $Asset) {
            throw "The manifest has no pinned release asset '$AssetKey'."
        }
        $Url = [string]$Asset.url
        if (-not $Url.StartsWith($AllowedPrefix)) {
            throw "Asset '$AssetKey' origin is not an official llama.cpp GitHub release URL."
        }

        # Cache keyed by pinned digest, so repeated or partially failed installs
        # never re-download bytes that were already verified once.
        $ZipPath = Join-Path $CacheDir ("{0}-{1}.zip" -f $AssetKey, $Asset.sha256)
        if ((Test-Path $ZipPath) -and ((Get-Item -LiteralPath $ZipPath).Length -eq [long]$Asset.size)) {
            Write-Host "Reusing cached download for $AssetKey."
        }
        else {
            Write-Host "Downloading $AssetKey from $Url"
            Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing
        }

        $Downloaded = Get-Item -LiteralPath $ZipPath
        if ($Downloaded.Length -ne [long]$Asset.size) {
            throw "Asset '$AssetKey' size mismatch: expected $($Asset.size) bytes, got $($Downloaded.Length)."
        }
        $Digest = Get-Sha256Hex -Path $ZipPath
        if ($Digest -ne [string]$Asset.sha256) {
            Remove-Item -LiteralPath $ZipPath -Force -ErrorAction SilentlyContinue
            throw "Asset '$AssetKey' SHA-256 mismatch; the cached copy was discarded."
        }

        $Archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
        try {
            foreach ($Entry in $Archive.Entries) {
                if ([string]::IsNullOrEmpty($Entry.Name)) { continue }
                if ($Entry.FullName -like "*..*") {
                    throw "Refusing unsafe archive path '$($Entry.FullName)'."
                }
                $IsExe = $Entry.Name.EndsWith(".exe")
                $IsDll = $Entry.Name.EndsWith(".dll")
                if (-not ($IsExe -or $IsDll)) {
                    throw "Unexpected archive entry '$($Entry.FullName)'; only executables and DLLs are accepted."
                }
                if ($IsExe -and ($RequiredExecutables -notcontains $Entry.Name)) {
                    throw "Unexpected executable entry '$($Entry.Name)'; not in requiredExecutables."
                }
                $Target = Join-Path $Destination $Entry.Name
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($Entry, $Target, $true)
            }
        }
        finally {
            $Archive.Dispose()
        }
        Write-Host "Verified and extracted $AssetKey."
    }
}
catch {
    throw $_
}

$Server = Join-Path $Destination "llama-server.exe"
if (-not (Test-Path $Server)) {
    throw "llama-server.exe was not found after extraction."
}
$VersionOutput = & $Server --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "llama-server.exe failed its --version executable health check."
}
Write-Host "llama-server installed at $Server"
Write-Host ($VersionOutput | Select-Object -First 1)
