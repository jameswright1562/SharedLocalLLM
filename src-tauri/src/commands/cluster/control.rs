use crate::{state::AppState, types::ClusterSession};

pub(crate) async fn halt(state: &AppState) {
    // Unload model from inference engine
    if let Some(inference) = state.inference.as_ref() {
        let _ = inference.unload().await;
    }

    // Stop any legacy external processes
    if let Ok(mut processes) = state.processes.lock() {
        processes.stop();
    }

    let mut peer = state.peer.lock().await;
    if let Some(forwarder) = peer.forwarder.take() {
        drop(peer);
        forwarder.shutdown().await;
        peer = state.peer.lock().await;
    }
    if let Some(client) = peer.client.clone() {
        drop(peer);
        let _ = client.stop_worker().await;
    }
    if let Some(server) = state.peer.lock().await.server.as_ref() {
        server.stop_local_worker().await;
    }
}

pub(crate) fn idle_session(has_peer: bool) -> ClusterSession {
    ClusterSession {
        status: if has_peer { "ready" } else { "idle" }.into(),
        ..ClusterSession::default()
    }
}

pub(super) async fn publish_cluster(state: &AppState) {
    let value = {
        let Ok(inner) = state.lock() else { return };
        let mut local = inner.local.clone();
        local.cluster_status = Some(inner.cluster.status.clone());
        local.cluster_model_id = inner.cluster.model_id.clone();
        serde_json::to_value(&local).ok()
    };
    if let (Some(value), Some(server)) = (value, state.peer.lock().await.server.as_ref()) {
        server.set_capabilities(value).await;
    }
}
