//! Windows Firewall rules for the Python peer backend.
//!
//! The Python sidecar owns TCP 49158 and UDP 49157, so rules are port-scoped
//! rather than executable-scoped. Network profile never blocks operation.

use std::path::Path;

pub const TCP_RULE_NAME: &str = "SharedLocalLLM";
pub const UDP_RULE_NAME: &str = "SharedLocalLLM Discovery";

pub async fn ensure_peer_firewall_rules(
    executable: &Path,
    tcp_port: u16,
    udp_port: u16,
) -> Result<(), String> {
    let _ = executable;
    #[cfg(windows)]
    {
        ensure_rule(TCP_RULE_NAME, "TCP", tcp_port).await?;
        ensure_rule(UDP_RULE_NAME, "UDP", udp_port).await
    }
    #[cfg(not(windows))]
    {
        let _ = (tcp_port, udp_port);
        Ok(())
    }
}

#[cfg(windows)]
async fn ensure_rule(name: &str, protocol: &str, port: u16) -> Result<(), String> {
    if rule_exists(name).await {
        return Ok(());
    }
    let script = firewall_rule_script(name, protocol, port);
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
fn firewall_rule_script(name: &str, protocol: &str, port: u16) -> String {
    format!(
        "New-NetFirewallRule -DisplayName {} -Direction Inbound -Action Allow -Protocol {} -LocalPort {} -Profile Any | Out-Null",
        powershell_literal(name), protocol, port,
    )
}

#[cfg(windows)]
fn powershell_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(windows)]
pub fn peer_firewall_rules_exist() -> bool {
    powershell_bool(
        "@(Get-NetFirewallRule -DisplayName 'SharedLocalLLM' -ErrorAction SilentlyContinue).Count -gt 0 -and @(Get-NetFirewallRule -DisplayName 'SharedLocalLLM Discovery' -ErrorAction SilentlyContinue).Count -gt 0",
    )
}

#[cfg(windows)]
pub fn is_elevated() -> bool {
    powershell_bool(
        "([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)",
    )
}

pub fn ensure_firewall_elevation() {
    #[cfg(windows)]
    {
        if peer_firewall_rules_exist() || is_elevated() {
            return;
        }
        let Ok(executable) = std::env::current_exe() else {
            return;
        };
        let launch = format!(
            "Start-Process -FilePath {} -Verb RunAs",
            powershell_literal(&executable.to_string_lossy())
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

    #[test]
    fn peer_rule_allows_tcp_port_on_every_profile() {
        let script = firewall_rule_script("SharedLocalLLM", "TCP", 49_158);
        assert!(script.contains("-Protocol TCP"));
        assert!(script.contains("-LocalPort 49158"));
        assert!(script.contains("-Profile Any"));
        assert!(!script.contains("-Program"));
    }

    #[test]
    fn discovery_rule_uses_udp() {
        let script = firewall_rule_script("SharedLocalLLM Discovery", "UDP", 49_157);
        assert!(script.contains("-Protocol UDP"));
        assert!(script.contains("-LocalPort 49157"));
    }
}
