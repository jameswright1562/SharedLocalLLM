//! Idempotent Windows Firewall rules for the peer channel.
//!
//! The peer channel listens on TCP port `49158` and broadcasts discovery on UDP
//! port `49157`. Windows labels unmanaged and directly-cabled (link-local)
//! networks as Public, so a Private-only rule would block direct Ethernet and
//! automatic `169.254.x.x` connections. These inbound rules permit only the
//! SharedLocalLLM executable on those two ports across every Windows profile
//! (Public, Private, Domain), making the network category irrelevant to
//! operation. They never disable Windows Firewall and never open a broad port.
//!
//! On startup SharedLocalLLM checks whether the rules already exist. If they
//! are missing and the process is not elevated, it relaunches itself with a UAC
//! prompt so the rules can be created; once present, later launches run without
//! elevation.

use std::path::Path;

pub const TCP_RULE_NAME: &str = "SharedLocalLLM";
pub const UDP_RULE_NAME: &str = "SharedLocalLLM Discovery";

/// Ensure program-scoped inbound rules exist for the peer TCP channel and UDP
/// discovery across all Windows profiles. Existing rules are kept untouched.
pub async fn ensure_peer_firewall_rules(
    executable: &Path,
    tcp_port: u16,
    udp_port: u16,
) -> Result<(), String> {
    #[cfg(windows)]
    {
        ensure_rule(TCP_RULE_NAME, "TCP", tcp_port, executable).await?;
        ensure_rule(UDP_RULE_NAME, "UDP", udp_port, executable).await
    }
    #[cfg(not(windows))]
    {
        let _ = (executable, tcp_port, udp_port);
        Ok(())
    }
}

#[cfg(windows)]
async fn ensure_rule(
    name: &str,
    protocol: &str,
    port: u16,
    executable: &Path,
) -> Result<(), String> {
    if rule_exists(name).await {
        return Ok(());
    }
    let script = firewall_rule_script(name, protocol, port, executable);
    let output = tokio::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .await
        .map_err(|error| format!("Windows Firewall configuration could not start: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(windows)]
async fn rule_exists(name: &str) -> bool {
    let script = format!(
        "@(Get-NetFirewallRule -DisplayName '{}' -ErrorAction SilentlyContinue).Count -gt 0",
        powershell_literal(name)
    );
    let Ok(output) = tokio::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .await
    else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .trim()
            .eq_ignore_ascii_case("True")
}

#[cfg(windows)]
fn firewall_rule_script(name: &str, protocol: &str, port: u16, executable: &Path) -> String {
    format!(
        "New-NetFirewallRule -DisplayName {} -Direction Inbound -Action Allow -Program {} -Protocol {} -LocalPort {} -Profile Any | Out-Null",
        powershell_literal(name),
        powershell_literal(&executable.to_string_lossy()),
        protocol,
        port,
    )
}

#[cfg(windows)]
fn powershell_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// True when both peer firewall rules already exist (no elevation needed).
#[cfg(windows)]
pub fn peer_firewall_rules_exist() -> bool {
    powershell_bool(
        "@(Get-NetFirewallRule -DisplayName 'SharedLocalLLM' -ErrorAction SilentlyContinue).Count -gt 0 -and @(Get-NetFirewallRule -DisplayName 'SharedLocalLLM Discovery' -ErrorAction SilentlyContinue).Count -gt 0",
    )
}

/// True when the current process holds an elevated Windows token.
#[cfg(windows)]
pub fn is_elevated() -> bool {
    powershell_bool(
        "([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)",
    )
}

/// Relaunch the process elevated if the peer firewall rules are missing and the
/// process is not already elevated. The elevated instance creates the rules
/// during peer-service startup; this instance exits immediately. If the user
/// declines the UAC prompt the app continues unprivileged and logs a warning
/// when rule creation is later attempted.
pub fn ensure_firewall_elevation() {
    #[cfg(windows)]
    {
        if peer_firewall_rules_exist() || is_elevated() {
            return;
        }
        let Ok(executable) = std::env::current_exe() else {
            return;
        };
        let exe = executable.to_string_lossy();
        let launch = format!(
            "Start-Process -FilePath {} -Verb RunAs",
            powershell_literal(&exe)
        );
        let relaunched = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", &launch])
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if relaunched {
            std::process::exit(0);
        }
    }
}

#[cfg(windows)]
fn powershell_bool(command: &str) -> bool {
    let Ok(output) = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", command])
        .output()
    else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .trim()
            .eq_ignore_ascii_case("True")
}

#[cfg(test)]
mod tests {
    use super::firewall_rule_script;
    use std::path::Path;

    #[test]
    fn inbound_rule_scopes_the_program_to_the_peer_port_across_all_profiles() {
        let script = firewall_rule_script(
            "SharedLocalLLM",
            "TCP",
            49_158,
            Path::new("C:\\Program Files\\SharedLocalLLM\\shared-local-llm.exe"),
        );
        assert!(script.contains("-Direction Inbound"));
        assert!(script.contains("-Action Allow"));
        assert!(script.contains("-Protocol TCP"));
        assert!(script.contains("-LocalPort 49158"));
        assert!(script.contains("-Profile Any"));
        assert!(
            script.contains("-Program 'C:\\Program Files\\SharedLocalLLM\\shared-local-llm.exe'")
        );
    }

    #[test]
    fn discovery_rule_uses_udp_and_quotes_single_quotes_in_paths() {
        let script = firewall_rule_script(
            "SharedLocalLLM Discovery",
            "UDP",
            49_157,
            Path::new("C:\\O'Brien\\shared-local-llm.exe"),
        );
        assert!(script.contains("-Protocol UDP"));
        assert!(script.contains("-LocalPort 49157"));
        assert!(script.contains("-Program 'C:\\O''Brien\\shared-local-llm.exe'"));
    }
}
