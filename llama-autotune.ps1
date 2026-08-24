param(
    [int]$PromptTokens = 20000,
    [int]$BroadRepetitions = 2,
    [int]$FinalRepetitions = 5,
    [switch]$RunIperf,
    [switch]$SkipDepthProfile,
    [switch]$FullDepthProfile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# USER CONFIG
# ============================================================

$LlamaDir = "C:\code\llama.cpp"

$BenchExe = Join-Path $LlamaDir "build\bin\Release\llama-bench.exe"

$ModelPath = "C:\Users\James\.cache\huggingface\hub\DavidAU\GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF\GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q8_0.gguf"

$RpcHost = "10.10.10.2"
$RpcPort = 50052
$RpcServer = "${RpcHost}:${RpcPort}"

# Used only when the script writes the recommended llama-server command.
$ServerHost = "10.10.10.1"
$ServerPort = 8080
$ServerContext = 160000

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ResultsDir = Join-Path $PSScriptRoot "llama-autotune-$Timestamp"

# ============================================================
# CANDIDATES
# ============================================================
# The script tunes prompt processing first because that is where
# your large OpenCode prompts were bottlenecking.

# Batch/ubatch are tested in valid pair groups to avoid useless
# combinations where ubatch > batch.
$BatchGroups = @(
    @{ Batch = "512,1024,2048"; UBatch = "256"  },
    @{ Batch = "512,1024,2048"; UBatch = "512"  },
    @{ Batch = "1024,2048";     UBatch = "1024" },
    @{ Batch = "2048";          UBatch = "2048" }
)

# First value applies to the first device printed by --list-devices,
# second value to the second device. Both directions are tested.
$TensorSplits = "40/60,45/55,50/50,55/45,60/40"

# Lower values intentionally included: with a 1 Gbps RPC link,
# keeping more work local/CPU can sometimes beat maximal remote offload.
$GpuLayers = "24,32,40,48,56,64,99"

$ThreadCounts = "4,6,8,12,16"

# q4/q8 cross-product = 4 tests.
# f16 is deliberately not auto-selected because a winner at 20k
# may not fit your real 160k server context.
$KTypes = "q4_0,q8_0"
$VTypes = "q4_0,q8_0"

$OpOffloadValues = "0,1"
$PollValues = "0,25,50,75,100"

# ============================================================
# CURRENT WINNING CONFIG
# ============================================================

$Current = [ordered]@{
    Batch       = 2048
    UBatch      = 512
    TensorSplit = $null
    GpuLayers   = 99
    Threads     = 8
    KType       = "q4_0"
    VType       = "q4_0"
    NoOpOffload = 0
    Poll        = 50
}

# Keep all rows so we can export one combined CSV at the end.
$AllResults = New-Object System.Collections.Generic.List[object]

# ============================================================
# HELPERS
# ============================================================

function Write-Section {
    param([string]$Text)

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Text
    Write-Host "============================================================"
}

function Safe-Name {
    param([string]$Name)
    return ($Name -replace '[^A-Za-z0-9_.-]', '_')
}

function Copy-CurrentConfig {
    $copy = [ordered]@{}
    foreach ($key in $Current.Keys) {
        $copy[$key] = $Current[$key]
    }
    return $copy
}

function New-BenchArguments {
    param(
        [hashtable]$Overrides,
        [int]$Repetitions
    )

    $cfg = Copy-CurrentConfig

    foreach ($key in $Overrides.Keys) {
        $cfg[$key] = $Overrides[$key]
    }

    $args = @(
        "-m", $ModelPath,
        "-rpc", $RpcServer,

        "-p", [string]$cfg.PromptTokens,
        "-n", [string]$cfg.GenTokens,
        "-d", [string]$cfg.Depth,

        "-ngl", [string]$cfg.GpuLayers,
        "-b", [string]$cfg.Batch,
        "-ub", [string]$cfg.UBatch,

        "-ctk", [string]$cfg.KType,
        "-ctv", [string]$cfg.VType,

        "-t", [string]$cfg.Threads,

        "-sm", "layer",
        "-fa", "on",

        "-nopo", [string]$cfg.NoOpOffload,
        "--poll", [string]$cfg.Poll,

        "-r", [string]$Repetitions,
        "-o", "json",
        "--progress"
    )

    if ($null -ne $cfg.TensorSplit -and [string]$cfg.TensorSplit -ne "") {
        $args += @("-ts", [string]$cfg.TensorSplit)
    }

    return $args
}

function Invoke-Bench {
    param(
        [string]$Stage,
        [hashtable]$Overrides,
        [int]$Repetitions = $BroadRepetitions
    )

    $safe = Safe-Name $Stage
    $jsonPath = Join-Path $ResultsDir "$safe.json"
    $errPath  = Join-Path $ResultsDir "$safe.stderr.log"

    if (-not $Overrides.ContainsKey("PromptTokens")) {
        $Overrides["PromptTokens"] = $PromptTokens
    }

    if (-not $Overrides.ContainsKey("GenTokens")) {
        $Overrides["GenTokens"] = 0
    }

    if (-not $Overrides.ContainsKey("Depth")) {
        $Overrides["Depth"] = 0
    }

    $arguments = New-BenchArguments `
        -Overrides $Overrides `
        -Repetitions $Repetitions

    Write-Host ""
    Write-Host "[$Stage]"
    Write-Host "$BenchExe $($arguments -join ' ')"
    Write-Host ""

    $raw = (& $BenchExe @arguments 2> $errPath | Out-String)
    $exitCode = $LASTEXITCODE

    $raw | Set-Content -Path $jsonPath -Encoding utf8

    if ($exitCode -ne 0) {
        Write-Warning "Stage '$Stage' failed with exit code $exitCode."
        Write-Warning "See: $errPath"
        return @()
    }

    try {
        $parsed = $raw | ConvertFrom-Json
    }
    catch {
        Write-Warning "Could not parse JSON from '$Stage'."
        Write-Warning "Raw output: $jsonPath"
        return @()
    }

    $rows = @($parsed)

    foreach ($row in $rows) {
        $row | Add-Member -NotePropertyName Stage -NotePropertyValue $Stage -Force
        $AllResults.Add($row)
    }

    $csvPath = Join-Path $ResultsDir "$safe.csv"
    $rows | Export-Csv -Path $csvPath -NoTypeInformation

    return $rows
}

function Get-PromptRows {
    param([object[]]$Rows)

    return @(
        $Rows |
            Where-Object {
                [int]$_.n_prompt -gt 0 -and
                [int]$_.n_gen -eq 0
            }
    )
}

function Get-GenerationRows {
    param([object[]]$Rows)

    return @(
        $Rows |
            Where-Object {
                [int]$_.n_prompt -eq 0 -and
                [int]$_.n_gen -gt 0
            }
    )
}

function Get-BestRow {
    param([object[]]$Rows)

    if (-not $Rows -or $Rows.Count -eq 0) {
        return $null
    }

    return $Rows |
        Sort-Object { [double]$_.avg_ts } -Descending |
        Select-Object -First 1
}

function Show-Rows {
    param(
        [object[]]$Rows,
        [string[]]$Properties
    )

    if (-not $Rows -or $Rows.Count -eq 0) {
        Write-Warning "No successful results."
        return
    }

    $Rows |
        Sort-Object { [double]$_.avg_ts } -Descending |
        Select-Object $Properties |
        Format-Table -AutoSize |
        Out-Host
}

function Convert-TensorSplitForServer {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    # llama-bench uses slash between devices; llama-server uses comma.
    return ($Value -replace '/', ',')
}

# ============================================================
# PRE-FLIGHT
# ============================================================

Write-Section "PRE-FLIGHT"

if (-not (Test-Path $BenchExe)) {
    throw "llama-bench.exe not found: $BenchExe"
}

if (-not (Test-Path $ModelPath)) {
    throw "Model not found: $ModelPath"
}

New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

Write-Host "Results directory:"
Write-Host $ResultsDir
Write-Host ""

Write-Host "Shutting down WSL..."
wsl.exe --shutdown
Start-Sleep -Seconds 2

$existingServer = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($existingServer) {
    Write-Host "Stopping local llama-server to free RAM/VRAM..."
    $existingServer | Stop-Process -Force
    Start-Sleep -Seconds 2
}

# ============================================================
# NETWORK CHECK
# ============================================================

Write-Section "NETWORK CHECK"

$adapters = Get-NetAdapter |
    Where-Object Status -eq "Up" |
    Select-Object Name, InterfaceDescription, LinkSpeed

$adapters | Format-Table -AutoSize | Out-Host
$adapters | Export-Csv `
    -Path (Join-Path $ResultsDir "network-adapters.csv") `
    -NoTypeInformation

$ethernet = $adapters |
    Where-Object Name -eq "Ethernet" |
    Select-Object -First 1

if ($ethernet) {
    if ([string]$ethernet.LinkSpeed -match '(^|\s)1\s*Gbps') {
        Write-Warning "Physical Ethernet is only 1 Gbps."
        Write-Warning "For two-machine llama.cpp RPC, this can be a major throughput bottleneck."
        Write-Warning "The 10 Gbps vEthernet/WSL adapter is virtual and is not the physical laptop link."
    }
}

Write-Host ""
Write-Host "Testing RPC worker $RpcServer..."

$rpcTest = Test-NetConnection `
    -ComputerName $RpcHost `
    -Port $RpcPort `
    -WarningAction SilentlyContinue

if (-not $rpcTest.TcpTestSucceeded) {
    throw "Cannot reach RPC worker at $RpcServer. Start ggml-rpc-server on the laptop first."
}

Write-Host "RPC worker reachable."

if ($RunIperf) {
    Write-Host ""
    Write-Host "Running iperf3 client test..."

    $iperf = Get-Command iperf3.exe -ErrorAction SilentlyContinue

    if (-not $iperf) {
        Write-Warning "iperf3.exe was not found in PATH. Skipping."
    }
    else {
        try {
            & $iperf.Source -c $RpcHost -P 4 -t 15 2>&1 |
                Tee-Object -FilePath (Join-Path $ResultsDir "iperf3.txt")
        }
        catch {
            Write-Warning "iperf3 failed. Make sure the laptop is running: iperf3.exe -s -B $RpcHost"
        }
    }
}

# ============================================================
# DEVICE ORDER
# ============================================================

Write-Section "LLAMA.CPP DEVICES"

$deviceOutput = (& $BenchExe -rpc $RpcServer --list-devices 2>&1 | Out-String)

$deviceOutput | Tee-Object `
    -FilePath (Join-Path $ResultsDir "devices.txt") |
    Out-Host

Write-Host ""
Write-Host "Tensor split numbers follow the device order shown above."
Write-Host "The script tests both sides of 50/50, so it does not need to guess which GPU is faster."

# Optional baseline GPU snapshot.
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    & $nvidiaSmi.Source 2>&1 |
        Set-Content -Path (Join-Path $ResultsDir "nvidia-smi-before.txt")
}

# ============================================================
# 1. BATCH / UBATCH
# ============================================================

Write-Section "1/7 - BATCH / UBATCH"

$batchRows = @()

$groupIndex = 0
foreach ($group in $BatchGroups) {
    $groupIndex++

    $rows = Invoke-Bench `
        -Stage "01-batch-group-$groupIndex" `
        -Overrides @{
            Batch  = $group.Batch
            UBatch = $group.UBatch
        }

    $batchRows += Get-PromptRows $rows
}

Show-Rows `
    -Rows $batchRows `
    -Properties @(
        "n_batch",
        "n_ubatch",
        "avg_ts",
        "stddev_ts"
    )

$best = Get-BestRow $batchRows

if ($best) {
    $Current.Batch  = [int]$best.n_batch
    $Current.UBatch = [int]$best.n_ubatch

    Write-Host ""
    Write-Host "WINNER: -b $($Current.Batch) -ub $($Current.UBatch) = $([math]::Round([double]$best.avg_ts, 2)) tok/s"
}

# ============================================================
# 2. TENSOR SPLIT
# ============================================================

Write-Section "2/7 - RPC TENSOR SPLIT"

$rows = Invoke-Bench `
    -Stage "02-tensor-split" `
    -Overrides @{
        TensorSplit = $TensorSplits
    }

$promptRows = Get-PromptRows $rows

Show-Rows `
    -Rows $promptRows `
    -Properties @(
        "tensor_split",
        "avg_ts",
        "stddev_ts"
    )

$best = Get-BestRow $promptRows

if ($best) {
    $Current.TensorSplit = [string]$best.tensor_split

    Write-Host ""
    Write-Host "WINNER: tensor split $($Current.TensorSplit) = $([math]::Round([double]$best.avg_ts, 2)) tok/s"
}

# ============================================================
# 3. GPU LAYERS
# ============================================================

Write-Section "3/7 - GPU LAYERS"

$rows = Invoke-Bench `
    -Stage "03-gpu-layers" `
    -Overrides @{
        GpuLayers = $GpuLayers
    }

$promptRows = Get-PromptRows $rows

Show-Rows `
    -Rows $promptRows `
    -Properties @(
        "n_gpu_layers",
        "avg_ts",
        "stddev_ts"
    )

$best = Get-BestRow $promptRows

if ($best) {
    $Current.GpuLayers = [int]$best.n_gpu_layers

    Write-Host ""
    Write-Host "WINNER: -ngl $($Current.GpuLayers) = $([math]::Round([double]$best.avg_ts, 2)) tok/s"
}

# ============================================================
# 4. CPU THREADS
# ============================================================

Write-Section "4/7 - CPU THREADS"

$rows = Invoke-Bench `
    -Stage "04-threads" `
    -Overrides @{
        Threads = $ThreadCounts
    }

$promptRows = Get-PromptRows $rows

Show-Rows `
    -Rows $promptRows `
    -Properties @(
        "n_threads",
        "avg_ts",
        "stddev_ts"
    )

$best = Get-BestRow $promptRows

if ($best) {
    $Current.Threads = [int]$best.n_threads

    Write-Host ""
    Write-Host "WINNER: -t $($Current.Threads) = $([math]::Round([double]$best.avg_ts, 2)) tok/s"
}

# ============================================================
# 5. KV CACHE FORMAT
# ============================================================

Write-Section "5/7 - KV CACHE FORMAT"

$rows = Invoke-Bench `
    -Stage "05-kv-cache" `
    -Overrides @{
        KType = $KTypes
        VType = $VTypes
    }

$promptRows = Get-PromptRows $rows

Show-Rows `
    -Rows $promptRows `
    -Properties @(
        "type_k",
        "type_v",
        "avg_ts",
        "stddev_ts"
    )

$best = Get-BestRow $promptRows

if ($best) {
    $Current.KType = [string]$best.type_k
    $Current.VType = [string]$best.type_v

    Write-Host ""
    Write-Host "WINNER: K=$($Current.KType) V=$($Current.VType) = $([math]::Round([double]$best.avg_ts, 2)) tok/s"
    Write-Warning "This winner was measured at $PromptTokens tokens. Always verify that it still fits your real 160k server context."
}

# ============================================================
# 6. OP OFFLOAD
# ============================================================

Write-Section "6/7 - OPERATION OFFLOAD"

$rows = Invoke-Bench `
    -Stage "06-op-offload" `
    -Overrides @{
        NoOpOffload = $OpOffloadValues
    }

$promptRows = Get-PromptRows $rows

Show-Rows `
    -Rows $promptRows `
    -Properties @(
        "no_op_offload",
        "avg_ts",
        "stddev_ts"
    )

$best = Get-BestRow $promptRows

if ($best) {
    $Current.NoOpOffload = [int]$best.no_op_offload

    Write-Host ""
    Write-Host "WINNER: no_op_offload=$($Current.NoOpOffload) = $([math]::Round([double]$best.avg_ts, 2)) tok/s"
}

# ============================================================
# 7. POLLING
# ============================================================

Write-Section "7/7 - CPU POLLING"

$rows = Invoke-Bench `
    -Stage "07-poll" `
    -Overrides @{
        Poll = $PollValues
    }

$promptRows = Get-PromptRows $rows

Show-Rows `
    -Rows $promptRows `
    -Properties @(
        "poll",
        "avg_ts",
        "stddev_ts"
    )

$best = Get-BestRow $promptRows

if ($best) {
    $Current.Poll = [int]$best.poll

    Write-Host ""
    Write-Host "WINNER: --poll $($Current.Poll) = $([math]::Round([double]$best.avg_ts, 2)) tok/s"
}

# ============================================================
# FINAL VERIFICATION: PP + TG
# ============================================================

Write-Section "FINAL VERIFICATION"

$finalRows = Invoke-Bench `
    -Stage "08-final-verification" `
    -Overrides @{
        PromptTokens = $PromptTokens
        GenTokens    = 256
        Depth        = 0
    } `
    -Repetitions $FinalRepetitions

Write-Host ""
Write-Host "Prompt processing:"
Show-Rows `
    -Rows (Get-PromptRows $finalRows) `
    -Properties @(
        "n_prompt",
        "n_batch",
        "n_ubatch",
        "type_k",
        "type_v",
        "n_gpu_layers",
        "n_threads",
        "tensor_split",
        "avg_ts",
        "stddev_ts"
    )

Write-Host ""
Write-Host "Token generation:"
Show-Rows `
    -Rows (Get-GenerationRows $finalRows) `
    -Properties @(
        "n_gen",
        "type_k",
        "type_v",
        "n_gpu_layers",
        "n_threads",
        "tensor_split",
        "avg_ts",
        "stddev_ts"
    )

# ============================================================
# GENERATION TENSOR-SPLIT CROSS-CHECK AT 20K DEPTH
# ============================================================

Write-Section "GENERATION SPLIT CROSS-CHECK AT 20K CONTEXT"

$generationSplitRows = Invoke-Bench `
    -Stage "09-generation-tensor-split" `
    -Overrides @{
        TensorSplit = $TensorSplits
        PromptTokens = 0
        GenTokens = 256
        Depth = 20000
    }

$generationRows = Get-GenerationRows $generationSplitRows

Show-Rows `
    -Rows $generationRows `
    -Properties @(
        "n_depth",
        "tensor_split",
        "avg_ts",
        "stddev_ts"
    )

$generationBest = Get-BestRow $generationRows

if ($generationBest) {
    Write-Host ""
    Write-Host "Best generation split at 20k depth: $($generationBest.tensor_split) = $([math]::Round([double]$generationBest.avg_ts, 2)) tok/s"

    if ([string]$generationBest.tensor_split -ne [string]$Current.TensorSplit) {
        Write-Warning "Prompt-processing and generation prefer different tensor splits."
        Write-Warning "The recommended server command below keeps the prompt-processing winner."
    }
}

# ============================================================
# DEPTH PROFILE
# ============================================================

if (-not $SkipDepthProfile) {
    Write-Section "CONTEXT DEPTH PROFILE"

    if ($FullDepthProfile) {
        $depthValues = "0,8192,20000,40000,80000,120000"
    }
    else {
        $depthValues = "0,8192,20000,40000"
    }

    $depthRows = Invoke-Bench `
        -Stage "10-context-depth-profile" `
        -Overrides @{
            PromptTokens = 512
            GenTokens    = 256
            Depth        = $depthValues
        }

    Show-Rows `
        -Rows $depthRows `
        -Properties @(
            "n_depth",
            "n_prompt",
            "n_gen",
            "avg_ts",
            "stddev_ts"
        )

    Write-Host ""
    Write-Host "Note: normal upstream llama-bench really computes the -d context prefill."
    Write-Host "Large depth values can therefore take a long time."
}

# ============================================================
# EXPORT EVERYTHING
# ============================================================

Write-Section "EXPORTING RESULTS"

$AllResults |
    Export-Csv `
        -Path (Join-Path $ResultsDir "all-results.csv") `
        -NoTypeInformation

$Current |
    ConvertTo-Json |
    Set-Content `
        -Path (Join-Path $ResultsDir "winning-config.json") `
        -Encoding utf8

# ============================================================
# WRITE RECOMMENDED SERVER COMMAND
# ============================================================

$serverTensorSplit = Convert-TensorSplitForServer `
    -Value ([string]$Current.TensorSplit)

$serverLines = New-Object System.Collections.Generic.List[string]

$serverLines.Add('$model = "' + $ModelPath + '"')
$serverLines.Add("")
$serverLines.Add('.\build\bin\Release\llama-server.exe `')
$serverLines.Add('  -m $model `')
$serverLines.Add("  --rpc $RpcServer ``")
$serverLines.Add("  -ngl $($Current.GpuLayers) ``")
$serverLines.Add("  -c $ServerContext ``")
$serverLines.Add("  -np 1 ``")
$serverLines.Add("  -ctk $($Current.KType) ``")
$serverLines.Add("  -ctv $($Current.VType) ``")
$serverLines.Add("  -fa on ``")
$serverLines.Add("  -b $($Current.Batch) ``")
$serverLines.Add("  -ub $($Current.UBatch) ``")
$serverLines.Add("  -t $($Current.Threads) ``")
$serverLines.Add("  --poll $($Current.Poll) ``")

if (-not [string]::IsNullOrWhiteSpace($serverTensorSplit)) {
    $serverLines.Add("  -ts $serverTensorSplit ``")
}

if ([int]$Current.NoOpOffload -eq 1) {
    $serverLines.Add("  --no-op-offload ``")
}

$serverLines.Add("  --jinja ``")
$serverLines.Add("  --reasoning-preserve ``")
$serverLines.Add("  --host $ServerHost ``")
$serverLines.Add("  --port $ServerPort")

$recommendedPath = Join-Path $ResultsDir "recommended-server-command.ps1"

$serverLines |
    Set-Content `
        -Path $recommendedPath `
        -Encoding utf8

# ============================================================
# FINAL SUMMARY
# ============================================================

Write-Section "AUTOTUNE COMPLETE"

Write-Host "Prompt-tuned winning configuration:"
Write-Host ""
Write-Host "  Batch        : $($Current.Batch)"
Write-Host "  UBatch       : $($Current.UBatch)"
Write-Host "  Tensor split : $($Current.TensorSplit)"
Write-Host "  GPU layers   : $($Current.GpuLayers)"
Write-Host "  Threads      : $($Current.Threads)"
Write-Host "  K cache      : $($Current.KType)"
Write-Host "  V cache      : $($Current.VType)"
Write-Host "  No op offload: $($Current.NoOpOffload)"
Write-Host "  Poll         : $($Current.Poll)"

Write-Host ""
Write-Host "Results:"
Write-Host "  $ResultsDir"

Write-Host ""
Write-Host "Recommended llama-server command:"
Write-Host "  $recommendedPath"

Write-Host ""
Write-Warning "Your physical laptop Ethernet link is 1 Gbps."
Write-Warning "If performance plateaus after tuning, the network link is one of the first hardware bottlenecks I would investigate."

Write-Host ""
Write-Host "CUDA build variants are NOT auto-built here."
Write-Host "For a fair FORCE_MMQ/FORCE_CUBLAS RPC comparison, both the desktop and laptop RPC worker should be rebuilt/tested consistently."