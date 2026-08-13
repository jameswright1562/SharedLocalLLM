use tauri::State;

use crate::{
    secrets,
    state::{data_root, regenerate_key, AppState},
    types::{ApiConfig, ErrorPayload},
};

#[tauri::command]
pub async fn get_api_config(state: State<'_, AppState>) -> Result<ApiConfig, ErrorPayload> {
    let (api_key, api_port) = {
        let inner = state.lock()?;
        (inner.api_key.clone(), inner.api_port)
    };
    let url = format!("http://127.0.0.1:{api_port}");
    let healthy = reqwest::Client::new()
        .get(format!("{url}/health"))
        .bearer_auth(&api_key)
        .timeout(std::time::Duration::from_millis(500))
        .send()
        .await
        .is_ok_and(|response| response.status().is_success());
    Ok(ApiConfig {
        url,
        api_key,
        healthy,
    })
}

#[tauri::command]
pub async fn regenerate_api_key(state: State<'_, AppState>) -> Result<ApiConfig, ErrorPayload> {
    let was_running = state.lock()?.cluster.status == "running";
    if was_running {
        state
            .processes
            .lock()
            .map_err(|_| {
                ErrorPayload::new(
                    "process_state",
                    "The runtime process manager is unavailable.",
                    None,
                )
            })?
            .stop();
        state.lock()?.cluster.status = "ready".into();
    }
    let key = regenerate_key();
    secrets::store(&data_root().join("secrets.dat"), key.as_bytes())?;
    state.lock()?.api_key = key;
    get_api_config(state).await
}
