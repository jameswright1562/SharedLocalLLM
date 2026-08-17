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
//! SharedLocalLLM runs elevated, so the rules are created with a direct
//! `New-NetFirewallRule` call and no UAC re-prompt is required.

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
