use crate::types::ErrorPayload;

const VALUE_NAME: &str = "SharedLocalLLM";

pub fn apply(enabled: bool) -> Result<(), ErrorPayload> {
    #[cfg(windows)]
    {
        apply_windows(enabled)
    }
    #[cfg(not(windows))]
    {
        let _ = enabled;
        Ok(())
    }
}

#[cfg(windows)]
fn apply_windows(enabled: bool) -> Result<(), ErrorPayload> {
    let exe = std::env::current_exe().map_err(autostart_error)?;
    let command = if enabled {
        format!(
            "New-ItemProperty -Path {} -Name {} -PropertyType String -Value {} -Force | Out-Null",
            powershell_literal(run_key()),
            powershell_literal(VALUE_NAME),
            powershell_literal(&exe.to_string_lossy())
        )
    } else {
        format!(
            "Remove-ItemProperty -Path {} -Name {} -ErrorAction SilentlyContinue",
            powershell_literal(run_key()),
            powershell_literal(VALUE_NAME)
        )
    };
    let output = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &command])
        .output()
        .map_err(autostart_error)?;
    if output.status.success() {
        Ok(())
    } else {
        Err(ErrorPayload::new(
            "autostart_failed",
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            Some("Start with Windows could not be updated.".into()),
        ))
    }
}

#[cfg(windows)]
fn run_key() -> &'static str {
    r"HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
}

#[cfg(windows)]
fn powershell_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(windows)]
fn autostart_error(error: std::io::Error) -> ErrorPayload {
    ErrorPayload::new("autostart_failed", error.to_string(), None)
}
