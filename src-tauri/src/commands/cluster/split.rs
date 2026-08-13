use tauri::State;

use crate::{
    capacity::{estimate_layer_split, DeviceLayerRequest, SplitEstimate, SplitEstimateRequest},
    state::AppState,
    types::{ErrorPayload, GpuLayerAllocation, ModelLoadConfig, ModelRecord, NodeCapabilities},
};

#[tauri::command]
pub fn estimate_model_split(
    model_id: String,
    load_config: ModelLoadConfig,
    state: State<'_, AppState>,
) -> Result<SplitEstimate, ErrorPayload> {
    let inner = state.lock()?;
    let model = inner
        .models
        .iter()
        .find(|model| model.id == model_id)
        .ok_or_else(model_not_found)?;
    let nodes = cluster_nodes(&inner.local, inner.peers.first());
    build_split_estimate(model, &load_config, &nodes).map(|(estimate, _)| estimate)
}

pub(super) fn cluster_nodes(
    local: &NodeCapabilities,
    peer: Option<&crate::pairing::PeerRecord>,
) -> Vec<NodeCapabilities> {
    let mut nodes = vec![local.clone()];
    if let Some(peer) = peer {
        nodes.push(
            peer.capabilities
                .clone()
                .unwrap_or_else(|| NodeCapabilities {
                    id: peer.id.clone(),
                    name: peer.name.clone(),
                    ..NodeCapabilities::default()
                }),
        );
    }
    nodes
}

pub(super) fn build_split_estimate(
    model: &ModelRecord,
    load_config: &ModelLoadConfig,
    nodes: &[NodeCapabilities],
) -> Result<(SplitEstimate, Vec<GpuLayerAllocation>), ErrorPayload> {
    let total_layers = model.layer_count.ok_or_else(|| {
        ErrorPayload::new(
            "layer_metadata_missing",
            "This GGUF file does not report a layer count for manual allocation.",
            Some("Use automatic allocation or refresh the model catalogue.".into()),
        )
    })?;
    let mut seen = std::collections::HashSet::new();
    for allocation in &load_config.gpu_layers {
        if !seen.insert(allocation.node_id.as_str()) {
            return Err(invalid_split(
                "A computer appears more than once in the GPU split.",
            ));
        }
        if !nodes.iter().any(|node| node.id == allocation.node_id) {
            return Err(invalid_split(
                "The GPU split includes a computer that is not in the active cluster.",
            ));
        }
    }
    let normalized = nodes
        .iter()
        .map(|node| GpuLayerAllocation {
            node_id: node.id.clone(),
            layers: load_config
                .gpu_layers
                .iter()
                .find(|allocation| allocation.node_id == node.id)
                .map(|allocation| allocation.layers)
                .unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    if normalized.iter().all(|allocation| allocation.layers == 0) {
        return Err(invalid_split(
            "Assign at least one model layer to a GPU, or use automatic allocation.",
        ));
    }
    let devices = nodes
        .iter()
        .zip(&normalized)
        .map(|(node, allocation)| {
            DeviceLayerRequest::new(
                &node.id,
                allocation.layers,
                (node.gpu.vram_available_gb.max(0.0) * 1024.0) as u64,
            )
        })
        .collect();
    let context_size = load_config
        .context_size
        .max(4096)
        .min(model.context_length.max(4096));
    let estimate = estimate_layer_split(SplitEstimateRequest {
        model_mib: model.size_bytes.div_ceil(1_048_576),
        total_layers,
        context_size,
        embedding_length: model.embedding_length,
        attention_head_count: model.attention_head_count,
        attention_head_count_kv: model.attention_head_count_kv,
        devices,
    })
    .map_err(|message| invalid_split(&message))?;
    Ok((estimate, normalized))
}

pub(super) fn distribute_layers_by_vram(
    total_layers: u32,
    nodes: &[NodeCapabilities],
) -> Vec<GpuLayerAllocation> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let reserved_per_node = u32::from(total_layers >= nodes.len() as u32);
    let distributable = total_layers.saturating_sub(reserved_per_node * nodes.len() as u32);
    let total_vram = nodes
        .iter()
        .map(|node| node.gpu.vram_available_gb.max(0.0))
        .sum::<f64>();
    let mut assigned = 0;
    nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let layers = if index == nodes.len().saturating_sub(1) {
                total_layers.saturating_sub(assigned)
            } else if total_vram > 0.0 {
                reserved_per_node
                    + ((distributable as f64 * node.gpu.vram_available_gb.max(0.0)) / total_vram)
                        .floor() as u32
            } else {
                reserved_per_node + distributable / nodes.len() as u32
            };
            assigned += layers;
            GpuLayerAllocation {
                node_id: node.id.clone(),
                layers,
            }
        })
        .collect()
}

fn model_not_found() -> ErrorPayload {
    ErrorPayload::new(
        "model_not_found",
        "Refresh the catalogue and choose an available model.",
        None,
    )
}

fn invalid_split(message: &str) -> ErrorPayload {
    ErrorPayload::new(
        "invalid_layer_split",
        message,
        Some("Adjust the per-computer GPU layer counts.".into()),
    )
}

#[cfg(test)]
mod tests {
    use super::distribute_layers_by_vram;
    use crate::types::{GpuInfo, NodeCapabilities};

    #[test]
    fn automatic_distribution_uses_both_gpus_and_preserves_every_layer() {
        let nodes = [
            NodeCapabilities {
                id: "local".into(),
                gpu: GpuInfo {
                    vram_available_gb: 14.0,
                    ..GpuInfo::default()
                },
                ..NodeCapabilities::default()
            },
            NodeCapabilities {
                id: "remote".into(),
                gpu: GpuInfo {
                    vram_available_gb: 8.0,
                    ..GpuInfo::default()
                },
                ..NodeCapabilities::default()
            },
        ];

        let split = distribute_layers_by_vram(40, &nodes);
        assert_eq!(split[0].layers, 25);
        assert_eq!(split[1].layers, 15);
        assert_eq!(split.iter().map(|item| item.layers).sum::<u32>(), 40);
    }
}
