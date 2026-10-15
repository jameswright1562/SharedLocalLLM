use crate::types::ErrorPayload;

const VALUE_NAME: &str = "SharedLocalLLM";

pub fn apply(enabled: bool) -> Result<(), ErrorPayload> {
    #[cfg(windows)]
    {
        apply_windows(enabled)
    }
    #[cfg(target_os = "macos")]
    {
        apply_macos(enabled)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
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

#[cfg(target_os = "macos")]
fn apply_macos(enabled: bool) -> Result<(), ErrorPayload> {
    use std::fs;
    let home = std::env::var("HOME")
        .map_err(|error| ErrorPayload::new("autostart_failed", error.to_string(), None))?;
    let agents = std::path::PathBuf::from(home)
        .join("Library")
        .join("LaunchAgents");
    let plist = agents.join("dev.sharedlocalllm.app.plist");
    if !enabled {
        if plist.exists() {
            fs::remove_file(&plist)
                .map_err(|error| ErrorPayload::new("autostart_failed", error.to_string(), None))?;
        }
        return Ok(());
    }
    let exe = std::env::current_exe()
        .map_err(|error| ErrorPayload::new("autostart_failed", error.to_string(), None))?;
    fs::create_dir_all(&agents).map_err(|error| {
        ErrorPayload::new(
            "autostart_failed",
            error.to_string(),
            Some("Check ~/Library/LaunchAgents permissions.".into()),
        )
    })?;
    let content = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.sharedlocalllm.app</string>
  <key>ProgramArguments</key><array><string>{}</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"#,
        exe.to_string_lossy().replace('&', "&amp;")
    );
    fs::write(&plist, content).map_err(|error| {
        ErrorPayload::new(
            "autostart_failed",
            error.to_string(),
            Some("Start at login could not be updated.".into()),
        )
    })?;
    Ok(())
}
