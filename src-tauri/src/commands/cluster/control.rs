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

    let (forwarders, clients) = {
        let mut peer = state.peer.lock().await;
        let legacy = peer.forwarder.take();
        let mut forwarders = std::mem::take(&mut peer.forwarders);
        if let Some(legacy) = legacy {
            forwarders.push(legacy);
        }
        let clients: Vec<_> = peer.clients.values().cloned().collect();
        let legacy_client = peer.client.clone();
        (forwarders, (clients, legacy_client))
    };
    for forwarder in forwarders {
        forwarder.shutdown().await;
    }
    // Ask every worker to stop (coordinator + up to 2 workers for the 3-node case).
    let mut seen = std::collections::HashSet::new();
    for client in clients.0.into_iter().chain(clients.1) {
        // Deduplicate by endpoint; PeerClient has no id, so use endpoint string.
        let key = client.endpoint().to_string();
        if seen.insert(key) {
            let _ = client.stop_worker().await;
        }
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
