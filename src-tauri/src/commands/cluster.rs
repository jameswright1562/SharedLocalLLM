use std::time::Duration;

use tauri::State;

pub mod benchmark;
pub mod split;

pub use benchmark::{cancel_inference_benchmark, run_inference_benchmark};
pub use split::estimate_model_split;

use crate::{
    commands::pairing::require_private_network,
    runtime,
    state::AppState,
    types::{ClusterSession, ErrorPayload, ModelLoadConfig},
};

#[tauri::command]
pub async fn start_cluster(
    model_id: String,
    load_config: ModelLoadConfig,
    state: State<'_, AppState>,
) -> Result<ClusterSession, ErrorPayload> {
    require_private_network()?;
    let (model, api_key, api_port, coordinator, peer_id, normalized_config) = {
        let inner = state.lock()?;
        let model = inner
            .models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
            .ok_or_else(model_not_found)?;
        let nodes = split::cluster_nodes(&inner.local, inner.peers.first());
        let mut normalized_config = ModelLoadConfig {
            context_size: load_config
                .context_size
                .max(4096)
                .min(model.context_length.max(4096)),
            gpu_layers: vec![],
        };
        if !load_config.gpu_layers.is_empty() {
            let (estimate, gpu_layers) = split::build_split_estimate(&model, &load_config, &nodes)?;
            if let Some(device) = estimate.devices.iter().find(|device| !device.fits) {
                let node_name = nodes
                    .iter()
                    .find(|node| node.id == device.node_id)
                    .map(|node| node.name.as_str())
                    .unwrap_or("A selected computer");
                return Err(ErrorPayload::new(
                    "split_exceeds_vram",
                    format!(
                        "{node_name} needs an estimated {} MiB of VRAM, but only {} MiB is currently available.",
                        device.estimated_vram_mib, device.available_vram_mib
                    ),
                    Some("Move layers to the other computer, reduce context, or use automatic allocation.".into()),
                ));
            }
            normalized_config.gpu_layers = gpu_layers;
        }
        (
            model,
            inner.api_key.clone(),
            inner.api_port,
            inner.local.id.clone(),
            inner.peers.first().map(|peer| peer.id.clone()),
            normalized_config,
        )
    };
    if runtime::status().status != "ready" {
        return Err(ErrorPayload::new(
            "runtime_missing",
            "The pinned llama.cpp runtime is not ready.",
            Some("Install it from the first-run setup.".into()),
        ));
    }
    state.log(
        "INFO",
        "cluster_start_requested",
        &format!(
            "topology={} context={} gpu_layers={}",
            if peer_id.is_some() {
                "distributed"
            } else {
                "local"
            },
            normalized_config.context_size,
            normalized_config
                .gpu_layers
                .iter()
                .map(|allocation| allocation.layers.to_string())
                .collect::<Vec<_>>()
                .join("/")
        ),
    );
    let rpc_endpoint = if peer_id.is_some() {
        let client = state.peer_client().await?;
        client.heartbeat().await?;
        let forwarder = client.start_rpc_forwarder().await?;
        let endpoint = forwarder.local_address().to_string();
        state.peer.lock().await.forwarder = Some(forwarder);
        Some(endpoint)
    } else {
        None
    };
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
        .start(
            &model,
            &normalized_config,
            &api_key,
            peer_id.is_some(),
            rpc_endpoint,
            api_port,
        )?;
    if let Err(error) = wait_for_health(api_port, &api_key).await {
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
        if let Some(forwarder) = state.peer.lock().await.forwarder.take() {
            forwarder.shutdown().await;
        }
        state.log("ERROR", "cluster_health_failed", &error.to_string());
        return Err(error);
    }
    let session = ClusterSession {
        status: "running".into(),
        coordinator_node_id: Some(coordinator),
        worker_node_id: peer_id,
        model_id: Some(model_id),
        error: None,
    };
    state.lock()?.cluster = session.clone();
    state.log(
        "INFO",
        "cluster_ready",
        "llama-server passed its loopback health check",
    );
    Ok(session)
}

fn model_not_found() -> ErrorPayload {
    ErrorPayload::new(
        "model_not_found",
        "Refresh the catalogue and choose an available model.",
        None,
    )
}

#[tauri::command]
pub async fn stop_cluster(state: State<'_, AppState>) -> Result<ClusterSession, ErrorPayload> {
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
    if let Some(forwarder) = state.peer.lock().await.forwarder.take() {
        forwarder.shutdown().await;
    }
    let mut inner = state.lock()?;
    inner.cluster = ClusterSession {
        status: if inner.peers.is_empty() {
            "idle"
        } else {
            "ready"
        }
        .into(),
        ..ClusterSession::default()
    };
    let session = inner.cluster.clone();
    drop(inner);
    state.log(
        "INFO",
        "cluster_stopped",
        "Stopped local runtime and encrypted RPC forwarding",
    );
    Ok(session)
}

async fn wait_for_health(port: u16, api_key: &str) -> Result<(), ErrorPayload> {
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
    Err(ErrorPayload::new(
        "llama_server_not_ready",
        "llama-server did not become healthy within ten seconds.",
        Some("Open the logs folder and check model/runtime compatibility.".into()),
    ))
}
