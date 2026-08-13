use std::time::Duration;

use crate::{
    state::AppState,
    types::{ClusterSession, ErrorPayload},
};

pub(crate) async fn halt(state: &AppState) {
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

pub(super) async fn wait_for_health(port: u16, api_key: &str) -> Result<(), ErrorPayload> {
    let url = format!("http://127.0.0.1:{port}/health");
    for _ in 0..40 {
        if reqwest::Client::new()
            .get(&url)
            .bearer_auth(api_key)
            .timeout(Duration::from_millis(500))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let log_tail = std::fs::read(
        dirs::data_local_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("SharedLocalLLM")
            .join("logs")
            .join("llama-server.stderr.log"),
    )
    .ok()
    .and_then(|bytes| {
        let start = bytes.len().saturating_sub(1500);
        String::from_utf8(bytes[start..].to_vec()).ok()
    })
    .filter(|text| !text.trim().is_empty());
    Err(ErrorPayload::new(
        "llama_server_not_ready",
        "llama-server did not become healthy within ten seconds.",
        Some(match log_tail {
            Some(tail) => format!(
                "Runtime log tail: {}",
                crate::state::redact_diagnostic(&tail)
            ),
            None => "Open the logs folder and check model/runtime compatibility.".into(),
        }),
    ))
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
