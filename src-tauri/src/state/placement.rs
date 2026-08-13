use crate::{
    capacity::{recommend_topology, CapacityOutcome, CapacityRequest},
    pairing::PeerRecord,
    types::{ModelRecord, NodeCapabilities},
};

pub(super) fn apply_fit(
    models: &mut [ModelRecord],
    local: &NodeCapabilities,
    peers: &[PeerRecord],
) {
    let mut gpu_mib = vec![(local.gpu.vram_available_gb.max(0.0) * 1024.0) as u64];
    gpu_mib.extend(
        peers
            .iter()
            .filter_map(|peer| peer.capabilities.as_ref())
            .map(|node| (node.gpu.vram_available_gb.max(0.0) * 1024.0) as u64),
    );
    let ram_mib = (local.ram_available_gb.max(0.0) * 1024.0) as u64;
    for model in models {
        let projector_mib = model
            .projector
            .as_ref()
            .and_then(|path| std::fs::metadata(path).ok())
            .map(|metadata| metadata.len().div_ceil(1_048_576))
            .unwrap_or_default();
        let context_mib = (model.context_length as u64 / 4096).max(1) * 512;
        let model_mib = model.size_bytes.div_ceil(1_048_576) + projector_mib + context_mib;
        model.fit =
            match recommend_topology(CapacityRequest::new(model_mib, gpu_mib.clone(), ram_mib))
                .outcome
            {
                CapacityOutcome::SingleNode { .. } => "single-node",
                CapacityOutcome::Distributed { .. } => "combined-gpu",
                CapacityOutcome::RamSpill { .. } => "gpu-ram",
                CapacityOutcome::Insufficient => "does-not-fit",
            }
            .into();
    }
}
