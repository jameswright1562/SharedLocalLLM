use std::path::{Path, PathBuf};

use crate::types::ErrorPayload;

pub(crate) fn require_private_network() -> Result<(), ErrorPayload> {
    #[cfg(windows)]
    {
        let profiles = active_network_profiles()?;
        if !network_profiles_are_trusted(&profiles) {
            return Err(private_network_error());
        }
    }
    Ok(())
}

#[cfg(windows)]
pub(super) fn require_pairing_network(allow_public_network: bool) -> Result<bool, ErrorPayload> {
    let profiles = active_network_profiles()?;
    if network_profiles_are_trusted(&profiles) {
        return Ok(false);
    }
    if !allow_public_network {
        return Err(private_network_error());
    }
    let confirmed = rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Warning)
        .set_title("Pair on a Public network?")
        .set_description("Only continue if you trust every device on this network. Your device name and pairing service may be discoverable for up to five minutes. Cluster launch remains blocked on a Windows Public network.")
        .set_buttons(rfd::MessageButtons::YesNo)
        .show() == rfd::MessageDialogResult::Yes;
    if !confirmed {
        return Err(ErrorPayload::new(
            "public_network_override_cancelled",
            "Pairing on the Public network was cancelled.",
            None,
        ));
    }
    Ok(true)
}

#[cfg(not(windows))]
pub(super) fn require_pairing_network(_allow_public_network: bool) -> Result<bool, ErrorPayload> {
    Ok(false)
}

pub(super) fn close_firewall_lease(lease: Option<&Path>) {
    if let Some(lease) = lease {
        let _ = std::fs::remove_file(lease);
    }
}

#[cfg(windows)]
pub(super) async fn open_temporary_public_firewall_port(
    port: u16,
) -> Result<PathBuf, ErrorPayload> {
    use base64::Engine as _;
    let id = uuid::Uuid::new_v4().to_string();
    let temporary_root = std::env::temp_dir().join("SharedLocalLLM");
    std::fs::create_dir_all(&temporary_root)
        .map_err(|error| ErrorPayload::new("public_firewall_temp", error.to_string(), None))?;
    let lease = temporary_root.join(format!("pairing-{id}.lease"));
    let ready = temporary_root.join(format!("pairing-{id}.ready"));
    std::fs::write(&lease, b"active")
        .map_err(|error| ErrorPayload::new("public_firewall_temp", error.to_string(), None))?;
    let executable = std::env::current_exe().map_err(|error| {
        ErrorPayload::new("public_firewall_executable", error.to_string(), None)
    })?;
    let rule_name = format!("SharedLocalLLM temporary pairing {id}");
    let script = format!(
        "$ErrorActionPreference='Stop'; New-NetFirewallRule -DisplayName {} -Direction Inbound -Action Allow -Program {} -Protocol TCP -LocalPort {} -Profile Public | Out-Null; Set-Content -LiteralPath {} -Value 'ready'; try {{ $deadline=(Get-Date).AddSeconds(300); while ((Test-Path -LiteralPath {}) -and ((Get-Date) -lt $deadline)) {{ Start-Sleep -Seconds 1 }} }} finally {{ Remove-NetFirewallRule -DisplayName {} -ErrorAction SilentlyContinue }}",
        powershell_literal(&rule_name), powershell_literal(&executable.to_string_lossy()), port,
        powershell_literal(&ready.to_string_lossy()), powershell_literal(&lease.to_string_lossy()), powershell_literal(&rule_name),
    );
    let encoded_bytes: Vec<u8> = script.encode_utf16().flat_map(u16::to_le_bytes).collect();
    let encoded = base64::engine::general_purpose::STANDARD.encode(encoded_bytes);
    let launch = format!("Start-Process -FilePath 'powershell.exe' -Verb RunAs -WindowStyle Hidden -ArgumentList @('-NoProfile','-NonInteractive','-WindowStyle','Hidden','-EncodedCommand','{encoded}')");
    let output = tokio::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &launch])
        .output()
        .await
        .map_err(|error| ErrorPayload::new("public_firewall_prompt", error.to_string(), None))?;
    if !output.status.success() {
        close_firewall_lease(Some(&lease));
        return Err(ErrorPayload::new(
            "public_firewall_denied",
            "Windows did not approve the temporary Public-network pairing rule.",
            Some("Approve the Windows prompt or change this network to Private.".into()),
        ));
    }
    for _ in 0..300 {
        if ready.is_file() {
            let _ = std::fs::remove_file(&ready);
            return Ok(lease);
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    close_firewall_lease(Some(&lease));
    Err(ErrorPayload::new(
        "public_firewall_timeout",
        "The temporary Public-network pairing rule was not created in time.",
        Some("Try again or change this network to Private.".into()),
    ))
}

#[cfg(not(windows))]
pub(super) async fn open_temporary_public_firewall_port(
    _port: u16,
) -> Result<PathBuf, ErrorPayload> {
    Err(ErrorPayload::new(
        "public_firewall_unsupported",
        "Temporary Public-network pairing is supported only on Windows.",
        None,
    ))
}

#[cfg(windows)]
fn powershell_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(windows)]
fn active_network_profiles() -> Result<String, ErrorPayload> {
    let output = std::process::Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", "@(Get-NetConnectionProfile | Where-Object IPv4Connectivity -ne 'Disconnected' | Select-Object -ExpandProperty NetworkCategory) -join ','"]).output()
        .map_err(|error| ErrorPayload::new("network_profile_probe", error.to_string(), None))?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(windows)]
fn private_network_error() -> ErrorPayload {
    ErrorPayload::new(
        "private_network_required",
        "This action requires a Windows Private network profile.",
        Some("Change the trusted LAN profile to Private in Windows Settings.".into()),
    )
}

fn network_profiles_are_trusted(profiles: &str) -> bool {
    profiles.split(',').any(|profile| {
        profile.trim().eq_ignore_ascii_case("Private")
            || profile.trim().eq_ignore_ascii_case("DomainAuthenticated")
    })
}

#[cfg(test)]
mod tests {
    use super::{network_profiles_are_trusted, temporary_firewall_script};
    use std::path::Path;
    #[test]
    fn accepts_private_and_domain_profiles_but_rejects_public_profiles() {
        assert!(network_profiles_are_trusted("Private"));
        assert!(network_profiles_are_trusted("Public,DomainAuthenticated"));
        assert!(!network_profiles_are_trusted("Public"));
        assert!(!network_profiles_are_trusted(""));
    }

    #[test]
    fn temporary_pairing_rule_allows_tcp_pairing_and_udp_discovery() {
        let script = temporary_firewall_script(
            "SharedLocalLLM temporary pairing test",
            Path::new("C:\\SharedLocalLLM.exe"),
            49_158,
            49_157,
            Path::new("C:\\ready"),
            Path::new("C:\\lease"),
        );
        assert!(script.contains("-Protocol TCP -LocalPort 49158"));
        assert!(script.contains("-Protocol UDP -LocalPort 49157"));
        assert!(script.contains("Remove-NetFirewallRule"));
    }
}
