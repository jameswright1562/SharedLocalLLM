use tauri::State;

pub mod benchmark;
mod control;
pub mod split;

pub use benchmark::{cancel_inference_benchmark, run_inference_benchmark};
pub(crate) use control::{halt as halt_runtime, idle_session as idle_cluster};
pub use split::estimate_model_split;

use crate::{
    state::AppState,
    types::{ClusterSession, ErrorPayload, ModelLoadConfig},
};

#[tauri::command]
pub async fn start_cluster(
    model_id: String,
    load_config: ModelLoadConfig,
    state: State<'_, AppState>,
) -> Result<ClusterSession, ErrorPayload> {
    let _guard = state.cluster_lock.lock().await;
    control::halt(&state).await;
    let (model, _api_key, _api_port, coordinator, peer_records, mut normalized_config) = {
        let inner = state.lock()?;
        let model = inner
            .models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
            .ok_or_else(model_not_found)?;
        if model.remote_only || model.shard_paths.is_empty() {
            return Err(ErrorPayload::new(
                "model_not_local",
                "This model file is on another computer.",
                Some(
                    "Launch it from the computer that stores the GGUF, or copy the file locally."
                        .into(),
                ),
            ));
        }
        let nodes = split::cluster_nodes(&inner.local, &inner.peers);
        let mut normalized_config = ModelLoadConfig {
            context_size: load_config
                .context_size
                .max(4096)
                .min(model.context_length.max(4096)),
            gpu_layers: load_config.gpu_layers.clone(),
            force: load_config.force,
        };
        if normalized_config.gpu_layers.is_empty() {
            if let Some(layers) = model.layer_count {
                normalized_config.gpu_layers = split::distribute_layers_by_vram(layers, &nodes);
            }
        }
        if !normalized_config.gpu_layers.is_empty() {
            let (estimate, gpu_layers) =
                split::build_split_estimate(&model, &normalized_config, &nodes)?;
            if !normalized_config.force {
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
            }
            normalized_config.gpu_layers = gpu_layers;
        }
        (
            model,
            inner.api_key.clone(),
            inner.api_port,
            inner.local.id.clone(),
            inner.peers.clone(),
            normalized_config,
        )
    };
    // Runtime check removed - using embedded llama.cpp via inference engine
    // if runtime::status().status != "ready" {
    //     return Err(ErrorPayload::new(
    //         "runtime_missing",
    //         "The pinned llama.cpp runtime is not ready.",
    //         Some("Install it from Settings or first-run setup.".into()),
    //     ));
    // }
    // Workers are peers with >0 assigned layers, or every online peer when the
    // model only fits across combined GPUs (supports 1 coordinator + up to 2 workers).
    let mut required_workers: Vec<String> = normalized_config
        .gpu_layers
        .iter()
        .filter(|allocation| {
            allocation.layers > 0
                && peer_records
                    .iter()
                    .any(|peer| peer.id == allocation.node_id)
        })
        .map(|allocation| allocation.node_id.clone())
        .collect();
    required_workers.sort();
    required_workers.dedup();
    if model.fit == "combined-gpu" && required_workers.is_empty() {
        required_workers = peer_records
            .iter()
            .filter(|peer| {
                peer.capabilities
                    .as_ref()
                    .is_none_or(|capabilities| capabilities.online)
            })
            .map(|peer| peer.id.clone())
            .collect();
    }
    let mut active_workers: Vec<String> = Vec::new();
    for worker_id in &required_workers {
        match state.peer_client_for(worker_id).await {
            Ok(client) if client.heartbeat().await.is_ok() => {
                active_workers.push(worker_id.clone());
            }
            _ if model.fit != "combined-gpu"
                && !normalized_config.gpu_layers.iter().any(|allocation| {
                    allocation.node_id == *worker_id && allocation.layers > 0
                }) =>
            {
                continue;
            }
            Ok(_) | Err(_) => {
                if model.fit == "combined-gpu" || !active_workers.is_empty() {
                    // Fall through to the combined error below for clarity.
                }
                normalized_config.gpu_layers.retain(|allocation| {
                    allocation.node_id != *worker_id
                        && !peer_records
                            .iter()
                            .any(|peer| peer.id == allocation.node_id)
                        || active_workers.contains(&allocation.node_id)
                });
                if model.fit == "combined-gpu" {
                    return Err(ErrorPayload::new(
                        "peer_unavailable",
                        "A paired computer did not answer, and this model needs all GPUs.",
                        Some(
                            "Bring every worker online or choose a model that fits locally.".into(),
                        ),
                    ));
                }
                state.log(
                    "WARN",
                    "peer_offline_local_launch",
                    "A paired computer is offline; launching without it",
                );
            }
        }
    }
    // If no worker is reachable but layers were assigned to peers, drop those
    // layers for a local-only fallback unless the model requires combined GPUs.
    if !required_workers.is_empty() && active_workers.is_empty() && model.fit != "combined-gpu" {
        let had_peer_layers = normalized_config
            .gpu_layers
            .iter()
            .any(|allocation| required_workers.contains(&allocation.node_id));
        if had_peer_layers {
            normalized_config
                .gpu_layers
                .retain(|allocation| !required_workers.contains(&allocation.node_id));
            state.log(
                "WARN",
                "peer_offline_local_launch",
                "The paired computers are offline; launching on this computer only",
            );
        }
    }
    let use_distributed = !active_workers.is_empty();
    state.log(
        "INFO",
        "cluster_start_requested",
        &format!(
            "topology={} workers={} context={} gpu_layers={}",
            if use_distributed {
                if active_workers.len() > 1 {
                    "distributed-3-node"
                } else {
                    "distributed"
                }
            } else {
                "local"
            },
            active_workers.len(),
            normalized_config.context_size,
            normalized_config
                .gpu_layers
                .iter()
                .map(|allocation| allocation.layers.to_string())
                .collect::<Vec<_>>()
                .join("/")
        ),
    );
    let rpc_endpoints = if use_distributed {
        let mut endpoints = Vec::new();
        for worker_id in &active_workers {
            let client = state.peer_client_for(worker_id).await?;
            client.heartbeat().await?;
            let forwarder = client.start_rpc_forwarder().await?;
            endpoints.push(forwarder.local_address().to_string());
            state.peer.lock().await.forwarders.push(forwarder);
        }
        endpoints
    } else {
        Vec::new()
    };
    // Use the embedded InferenceEngine
    if let Some(inference) = state.inference.as_ref() {
        let gpu_layers = normalized_config.gpu_layers.clone();

        state.log(
            "DEBUG",
            "loading_model_embedded",
            &format!(
                "Loading {} with {} GPU layers across {} worker(s)",
                model.id,
                gpu_layers.iter().map(|a| a.layers).sum::<u32>(),
                active_workers.len()
            ),
        );

        inference
            .load_multi(
                model.shard_paths[0].clone(),
                normalized_config.context_size,
                gpu_layers,
                rpc_endpoints,
            )
            .await?;

        state.log(
            "INFO",
            "model_loaded_embedded",
            "Model loaded successfully in the embedded inference engine",
        );
    } else {
        return Err(ErrorPayload::new(
            "inference_unavailable",
            "The embedded inference engine failed to initialize.",
            Some("Check the application logs for details.".into()),
        ));
    }
    let session = ClusterSession {
        status: "running".into(),
        coordinator_node_id: Some(coordinator),
        worker_node_id: active_workers.first().cloned(),
        worker_node_ids: if active_workers.is_empty() {
            None
        } else {
            Some(active_workers.clone())
        },
        model_id: Some(model_id),
        error: None,
    };
    state.lock()?.cluster = session.clone();
    control::publish_cluster(&state).await;
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
    let _guard = state.cluster_lock.lock().await;
    control::halt(&state).await;
    let session = {
        let mut inner = state.lock()?;
        inner.cluster = control::idle_session(!inner.peers.is_empty());
        inner.cluster.clone()
    };
    control::publish_cluster(&state).await;
    state.log(
        "INFO",
        "cluster_stopped",
        "Stopped local runtime, remote worker, and encrypted RPC forwarding",
    );
    Ok(session)
}
