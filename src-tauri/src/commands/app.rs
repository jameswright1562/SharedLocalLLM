use std::fs;

use tauri::{Emitter, State};

use crate::{
    hardware, runtime,
    state::{logs_root, AppState},
    types::{AppSettings, AppSnapshot, ErrorPayload},
};

#[tauri::command]
pub async fn get_app_snapshot(state: State<'_, AppState>) -> Result<AppSnapshot, ErrorPayload> {
    crate::commands::pairing::lifecycle::refresh_peer_status(&state).await;
    merge_peer_catalogue(&state).await;
    state.snapshot()
}

async fn merge_peer_catalogue(state: &AppState) {
    let Ok(client) = state.peer_client().await else {
        return;
    };
    let Ok(value) = client.remote_models().await else {
        return;
    };
    let Ok(mut remote) = serde_json::from_value::<Vec<crate::types::ModelRecord>>(value) else {
        return;
    };
    for model in &mut remote {
        model.remote_only = model.shard_paths.is_empty();
        if model.locations.is_empty() {
            continue;
        }
    }
    if let Ok(mut inner) = state.lock() {
        let local_ids = inner
            .models
            .iter()
            .map(|model| model.id.clone())
            .collect::<std::collections::HashSet<_>>();
        for model in remote {
            if !local_ids.contains(&model.id) {
                inner.models.push(model);
            }
        }
    }
    if let Ok(models) = state.lock().map(|inner| inner.models.clone()) {
        if let Some(server) = state.peer.lock().await.server.as_ref() {
            if let Ok(value) = serde_json::to_value(
                models
                    .iter()
                    .filter(|model| !model.remote_only)
                    .collect::<Vec<_>>(),
            ) {
                server.set_catalogue(value).await;
            }
        }
    }
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
pub async fn update_settings(
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
    crate::autostart::apply(settings.autostart)?;
    let (current_port, cluster_running, api_key) = {
        let inner = state.lock()?;
        (
            inner.api_port,
            inner.cluster.status == "running",
            inner.api_key.clone(),
        )
    };
    if settings.api_port != current_port
        && std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, settings.api_port)).is_err()
    {
        return Err(ErrorPayload::new(
            "api_port_in_use",
            format!("127.0.0.1:{} is already in use.", settings.api_port),
            Some("Choose another local API port.".into()),
        ));
    }
    if cluster_running && settings.api_port != current_port {
        crate::commands::cluster::halt_runtime(&state).await;
        let mut inner = state.lock()?;
        inner.cluster = crate::commands::cluster::idle_cluster(!inner.peers.is_empty());
    }
    {
        let mut inner = state.lock()?;
        inner.local.name = name.into();
        inner.api_port = settings.api_port;
        inner.autostart = settings.autostart;
    }
    if let Some(server) = state.peer.lock().await.server.as_ref() {
        server.set_api(api_key, settings.api_port).await;
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
pub async fn refresh_hardware(state: State<'_, AppState>) -> Result<AppSnapshot, ErrorPayload> {
    let (id, name) = {
        let inner = state.lock()?;
        (inner.local.id.clone(), inner.local.name.clone())
    };
    let mut local = hardware::probe_local();
    local.id = id;
    local.name = name;
    state.lock()?.local = local;
    crate::commands::pairing::lifecycle::refresh_peer_status(&state).await;
    state.refresh_models_shared()?;
    state.log(
        "INFO",
        "hardware_refreshed",
        "Updated local and peer capabilities",
    );
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
