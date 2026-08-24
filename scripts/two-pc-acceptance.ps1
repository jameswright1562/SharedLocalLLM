<#
.SYNOPSIS
    Two-PC hardware acceptance smoke checks for SharedLocalLLM.

.DESCRIPTION
    Verifies firewall rules, listening peer/discovery endpoints, TCP reachability of the peer
    computer, and that any running llama.cpp sidecars are loopback-only. These are
    network/firewall/loopback smoke checks, not proof of distributed GPU inference; the manual
    two-computer acceptance matrix in docs/testing.md remains authoritative.

.PARAMETER PeerAddress
    IPv4 address of the peer PC. Defaults to 10.10.10.2 (static direct Ethernet).

.EXAMPLE
    .\scripts\two-pc-acceptance.ps1 -PeerAddress 10.10.10.2
#>
param(
    [string]$PeerAddress = '10.10.10.2'
)

$ErrorActionPreference = 'Stop'

$PeerPort = 49158
$DiscoveryPort = 49157
$checks = @()

function Add-Check {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Detail
    )
    $script:checks += [pscustomobject]@{
        Check  = $Name
        Status = $Status
        Detail = $Detail
    }
}

foreach ($rule in @(
        @{ Name = 'SharedLocalLLM Peer Backend'; Protocol = 'TCP'; Port = $PeerPort },
        @{ Name = 'SharedLocalLLM Peer Discovery'; Protocol = 'UDP'; Port = $DiscoveryPort }
    )) {
    try {
        $firewall = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction Stop |
            Select-Object -First 1
        if ($null -eq $firewall) {
            Add-Check "Firewall rule $($rule.Name)" 'FAIL' 'Rule not found'
            continue
        }
        $issues = @()
        if ($firewall.Enabled -ne $true) {
            $issues += "Enabled=$($firewall.Enabled)"
        }
        if ("$($firewall.Direction)" -ne 'Inbound') {
            $issues += "Direction=$($firewall.Direction)"
        }
        if ("$($firewall.Action)" -ne 'Allow') {
            $issues += "Action=$($firewall.Action)"
        }
        if ("$($firewall.Profile)" -ne 'Any') {
            $issues += "Profile=$($firewall.Profile)"
        }
        if ($issues.Count -eq 0) {
            Add-Check "Firewall rule $($rule.Name)" 'PASS' 'Enabled, Inbound, Allow, Profile=Any'
        } else {
            Add-Check "Firewall rule $($rule.Name)" 'FAIL' ($issues -join '; ')
        }
    } catch {
        Add-Check "Firewall rule $($rule.Name)" 'FAIL' $_.Exception.Message
    }
}

try {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $PeerPort -ErrorAction Stop)
    $bound = @($listeners | Where-Object {
            $address = "$($_.LocalAddress)"
            $address -and $address -ne '127.0.0.1' -and $address -ne '::1'
        })
    if ($bound.Count -gt 0) {
        $detail = ($bound | ForEach-Object { "$($_.LocalAddress):$($_.LocalPort)" } |
            Sort-Object -Unique) -join ', '
        Add-Check "Peer channel listening (TCP $PeerPort)" 'PASS' $detail
    } else {
        $detail = ($listeners | ForEach-Object { "$($_.LocalAddress):$($_.LocalPort)" }) -join ', '
        if ($detail) {
            Add-Check "Peer channel listening (TCP $PeerPort)" 'FAIL' "No non-loopback listener; found: $detail"
        } else {
            Add-Check "Peer channel listening (TCP $PeerPort)" 'FAIL' 'No listening connection on port 49158'
        }
    }
} catch {
    Add-Check "Peer channel listening (TCP $PeerPort)" 'FAIL' $_.Exception.Message
}

try {
    $udp = @(Get-NetUDPEndpoint -LocalPort $DiscoveryPort -ErrorAction Stop)
    if ($udp.Count -gt 0) {
        Add-Check "Discovery endpoint (UDP $DiscoveryPort)" 'PASS' "Bound endpoint(s): $($udp.Count)"
    } else {
        Add-Check "Discovery endpoint (UDP $DiscoveryPort)" 'FAIL' 'No UDP endpoint bound'
    }
} catch {
    Add-Check "Discovery endpoint (UDP $DiscoveryPort)" 'FAIL' $_.Exception.Message
}

try {
    $reached = Test-NetConnection -ComputerName $PeerAddress -Port $PeerPort `
        -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($reached -eq $true) {
        Add-Check "TCP reachability $PeerAddress`:$PeerPort" 'PASS' 'TcpTestSucceeded=True'
    } else {
        Add-Check "TCP reachability $PeerAddress`:$PeerPort" 'FAIL' 'TcpTestSucceeded=False'
    }
} catch {
    Add-Check "TCP reachability $PeerAddress`:$PeerPort" 'FAIL' $_.Exception.Message
}

$sidecars = @(
    Get-Process -Name 'sharedlocalllm-backend', 'ggml-rpc-server', 'llama-server' `
        -ErrorAction SilentlyContinue
)
if ($sidecars.Count -eq 0) {
    Add-Check 'Loopback-only sidecars' 'SKIP' 'cluster not running - loopback check skipped'
} else {
    $names = ($sidecars | ForEach-Object { $_.ProcessName } | Sort-Object -Unique) -join ', '
    $violations = @()
    foreach ($process in $sidecars) {
        $owned = @(Get-NetTCPConnection -State Listen -OwningProcess $process.Id `
                -ErrorAction SilentlyContinue)
        foreach ($connection in $owned) {
            $address = "$($connection.LocalAddress)"
            $isPeerListener = $process.ProcessName -eq 'sharedlocalllm-backend' -and `
                $connection.LocalPort -eq $PeerPort
            if (-not $isPeerListener -and $address -and $address -ne '127.0.0.1' -and $address -ne '::1') {
                $violations += "$($process.ProcessName) pid $($process.Id) listens on $address`:$($connection.LocalPort)"
            }
        }
    }
    if ($violations.Count -eq 0) {
        Add-Check 'Loopback-only sidecars' 'PASS' "$names listeners are loopback-only"
    } else {
        Add-Check 'Loopback-only sidecars' 'FAIL' ($violations -join '; ')
    }
}

Write-Host ''
Write-Host "Peer acceptance smoke checks against $PeerAddress" -ForegroundColor Cyan
Write-Host ('-' * 60)
$checks | Format-Table -AutoSize Check, Status, Detail
$failed = @($checks | Where-Object { $_.Status -eq 'FAIL' })
if ($failed.Count -gt 0) {
    Write-Host "RESULT: FAIL ($($failed.Count) check(s) failed)" -ForegroundColor Red
    exit 1
} else {
    Write-Host 'RESULT: PASS' -ForegroundColor Green
    exit 0
}
