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
        peer.clients.clear();
        (
            peer.forwarder.take(),
            std::mem::take(&mut peer.forwarders),
            peer.discovery.take(),
            peer.server.take(),
        )
    };
    if let Some(forwarder) = runtime.0 {
        forwarder.shutdown().await;
    }
    for forwarder in runtime.1 {
        forwarder.shutdown().await;
    }
    if let Some(discovery) = runtime.2 {
        discovery.shutdown().await;
    }
    if let Some(server) = runtime.3 {
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
        "Removed all paired computers; model files were not changed",
    );
    lifecycle::start_persistent_peer_service(app).await;
    state.snapshot()
}

/// Forget one peer (for the 3-node case) without dropping the other computer.
#[tauri::command]
pub async fn forget_peer(
    peer_id: String,
    state: State<'_, AppState>,
) -> Result<AppSnapshot, ErrorPayload> {
    {
        let mut peer = state.peer.lock().await;
        peer.clients.remove(&peer_id);
        if peer.clients.is_empty() {
            peer.client = None;
        }
    }
    {
        let mut inner = state.lock()?;
        inner.peers.retain(|peer| peer.id != peer_id);
        if inner.cluster.workers().contains(&peer_id) {
            inner.cluster = ClusterSession::default();
        }
    }
    state.persist()?;
    if let Err(error) = state.refresh_models_shared() {
        state.log("WARN", "model_fit_refresh_failed", &error.to_string());
    }
    state.log(
        "INFO",
        "peer_forgotten",
        "Forgot one paired computer; model files were not changed",
    );
    state.snapshot()
}
