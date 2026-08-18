use tauri::{AppHandle, State};

use crate::{
    state::AppState,
    types::{AppSnapshot, ClusterSession, ErrorPayload},
};

use super::lifecycle;

#[tauri::command]
pub async fn reset_pairing(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<AppSnapshot, ErrorPayload> {
    state
        .processes
        .lock()
        .map_err(|_| ErrorPayload::new("process_state", "Runtime state is unavailable.", None))?
        .stop();
    let runtime = {
        let mut peer = state.peer.lock().await;
        peer.client = None;
        (
            peer.forwarder.take(),
            peer.discovery.take(),
            peer.server.take(),
        )
    };
    if let Some(forwarder) = runtime.0 {
        forwarder.shutdown().await;
    }
    if let Some(discovery) = runtime.1 {
        discovery.shutdown().await;
    }
    if let Some(server) = runtime.2 {
        server.shutdown().await;
    }
    {
        let mut inner = state.lock()?;
        inner.peers.clear();
        inner.network = None;
        inner.cluster = ClusterSession::default();
    }
    state.persist()?;
    if let Err(error) = state.refresh_models_shared() {
        state.log("WARN", "model_fit_refresh_failed", &error.to_string());
    }
    state.log(
        "INFO",
        "pairing_reset",
        "Removed the connected peer; model files were not changed",
    );
    lifecycle::start_persistent_peer_service(app).await;
    state.snapshot()
}
