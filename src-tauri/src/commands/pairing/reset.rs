use tauri::{AppHandle, State};

use crate::{
    secrets,
    state::{peer_secret_path, AppState},
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
        peer.pairing_session_id = None;
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
    let peer_ids = {
        let mut inner = state.lock()?;
        let ids = inner
            .peers
            .iter()
            .map(|peer| peer.id.clone())
            .collect::<Vec<_>>();
        inner.peers.clear();
        inner.network = None;
        inner.cluster = ClusterSession::default();
        ids
    };
    state.persist()?;
    if let Err(error) = state.refresh_models_shared() {
        state.log("WARN", "model_fit_refresh_failed", &error.to_string());
    }
    for peer_id in peer_ids {
        if let Err(error) = secrets::remove(&peer_secret_path(&peer_id)) {
            state.log("WARN", "peer_secret_cleanup_failed", &error.to_string());
        }
    }
    state.log(
        "INFO",
        "pairing_reset",
        "Removed peer trust; model files were not changed",
    );
    lifecycle::start_persistent_peer_service(app).await;
    state.snapshot()
}
