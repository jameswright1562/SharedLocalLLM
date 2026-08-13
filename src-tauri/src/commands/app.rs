use std::fs;

use tauri::{Emitter, State};

use crate::{
    hardware, runtime,
    state::{logs_root, AppState},
    types::{AppSettings, AppSnapshot, ErrorPayload},
};

#[tauri::command]
pub fn get_app_snapshot(state: State<'_, AppState>) -> Result<AppSnapshot, ErrorPayload> {
    state.snapshot()
}

#[tauri::command]
pub fn complete_setup(
    device_name: String,
    state: State<'_, AppState>,
) -> Result<AppSnapshot, ErrorPayload> {
    let name = device_name.trim();
    if name.is_empty() || name.len() > 80 {
        return Err(ErrorPayload::new(
            "device_name_invalid",
            "Device name must contain between 1 and 80 characters.",
            None,
        ));
    }
    {
        let mut inner = state.lock()?;
        inner.local.name = name.into();
        inner.setup_complete = true;
    }
    state.persist()?;
    state.snapshot()
}

#[tauri::command]
pub fn update_settings(
    settings: AppSettings,
    state: State<'_, AppState>,
) -> Result<AppSnapshot, ErrorPayload> {
    let name = settings.device_name.trim();
    if name.is_empty() || name.len() > 80 || settings.api_port < 1024 {
        return Err(ErrorPayload::new(
            "settings_invalid",
            "Use a device name of 1-80 characters and an API port from 1024-65535.",
            None,
        ));
    }
    let current_port = state.lock()?.api_port;
    if settings.api_port != current_port
        && std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, settings.api_port)).is_err()
    {
        return Err(ErrorPayload::new(
            "api_port_in_use",
            format!("127.0.0.1:{} is already in use.", settings.api_port),
            Some("Choose another local API port.".into()),
        ));
    }
    {
        let mut inner = state.lock()?;
        inner.local.name = name.into();
        inner.api_port = settings.api_port;
        inner.autostart = settings.autostart;
    }
    state.persist()?;
    state.snapshot()
}

#[tauri::command]
pub async fn install_runtime(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<AppSnapshot, ErrorPayload> {
    runtime::install(|progress| {
        let _ = app.emit("runtime-progress", progress);
    })
    .await?;
    state.snapshot()
}

#[tauri::command]
pub fn refresh_hardware(state: State<'_, AppState>) -> Result<AppSnapshot, ErrorPayload> {
    state.lock()?.local = hardware::probe_local();
    state.refresh_models_shared()?;
    state.snapshot()
}

#[tauri::command]
pub fn open_logs_folder() -> Result<(), ErrorPayload> {
    let path = logs_root();
    fs::create_dir_all(&path)
        .map_err(|error| ErrorPayload::new("logs_folder", error.to_string(), None))?;
    #[cfg(windows)]
    std::process::Command::new("explorer.exe")
        .arg(&path)
        .spawn()
        .map_err(|error| ErrorPayload::new("logs_folder", error.to_string(), None))?;
    Ok(())
}

#[tauri::command]
pub fn open_network_settings() -> Result<(), ErrorPayload> {
    #[cfg(windows)]
    {
        std::process::Command::new("explorer.exe")
            .arg("ms-settings:network-status")
            .spawn()
            .map_err(|error| {
                ErrorPayload::new(
                    "network_settings",
                    error.to_string(),
                    Some("Open Settings > Network & internet manually.".into()),
                )
            })?;
        Ok(())
    }
    #[cfg(not(windows))]
    Err(ErrorPayload::new(
        "network_settings_unsupported",
        "Windows network settings are unavailable on this operating system.",
        None,
    ))
}
