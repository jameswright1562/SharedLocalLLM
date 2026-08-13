use serde::{Deserialize, Serialize};

pub const GPU_MARGIN_MIB: u64 = 1024;
pub const DEFAULT_CONTEXT: u32 = 8192;
pub const MIN_CONTEXT: u32 = 4096;
pub const MIN_RAM_RESERVE_MIB: u64 = 6 * 1024;
pub const GPU_RUNTIME_ALLOWANCE_MIB: u64 = 512;

#[derive(Clone, Debug)]
pub struct CapacityRequest {
    pub model_mib: u64,
    pub gpu_mib: Vec<u64>,
    pub system_ram_mib: u64,
    pub requested_context: Option<u32>,
}

impl CapacityRequest {
    pub fn new(model_mib: u64, gpu_mib: Vec<u64>, system_ram_mib: u64) -> Self {
        Self {
            model_mib,
            gpu_mib,
            system_ram_mib,
            requested_context: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CapacityOutcome {
    SingleNode { gpu_index: usize },
    Distributed { splits: Vec<f64> },
    RamSpill { splits: Vec<f64>, spill_mib: u64 },
    Insufficient,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CapacityRecommendation {
    pub outcome: CapacityOutcome,
    pub context_size: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DeviceLayerRequest {
    pub node_id: String,
    pub layers: u32,
    pub available_vram_mib: u64,
}

impl DeviceLayerRequest {
    pub fn new(node_id: impl Into<String>, layers: u32, available_vram_mib: u64) -> Self {
        Self {
            node_id: node_id.into(),
            layers,
            available_vram_mib,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SplitEstimateRequest {
    pub model_mib: u64,
    pub total_layers: u32,
    pub context_size: u32,
    pub embedding_length: Option<u32>,
    pub attention_head_count: Option<u32>,
    pub attention_head_count_kv: Option<u32>,
    pub devices: Vec<DeviceLayerRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DeviceVramEstimate {
    pub node_id: String,
    pub layers: u32,
    pub estimated_vram_mib: u64,
    pub available_vram_mib: u64,
    pub fits: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SplitEstimate {
    pub total_layers: u32,
    pub gpu_layers: u32,
    pub cpu_layers: u32,
    pub estimated_cpu_ram_mib: u64,
    pub uses_attention_metadata: bool,
    pub devices: Vec<DeviceVramEstimate>,
}

pub fn estimate_layer_split(request: SplitEstimateRequest) -> Result<SplitEstimate, String> {
    if request.total_layers == 0 {
        return Err("The model does not report a usable layer count.".into());
    }
    let gpu_layers = request
        .devices
        .iter()
        .try_fold(0u32, |total, device| total.checked_add(device.layers))
        .ok_or_else(|| "The selected GPU layer count overflowed.".to_owned())?;
    if gpu_layers > request.total_layers {
        return Err(format!(
            "The split selects {gpu_layers} GPU layers, but the model has {}.",
            request.total_layers
        ));
    }

    let kv_per_layer_mib = kv_cache_mib_per_layer(&request);
    let uses_attention_metadata = request.embedding_length.is_some()
        && request.attention_head_count.is_some()
        && request.attention_head_count_kv.is_some();
    let devices = request
        .devices
        .into_iter()
        .map(|device| {
            let weight_mib =
                proportional_mib(request.model_mib, device.layers, request.total_layers);
            let estimated_vram_mib = if device.layers == 0 {
                0
            } else {
                weight_mib
                    .saturating_add(kv_per_layer_mib.saturating_mul(device.layers as u64))
                    .saturating_add(GPU_RUNTIME_ALLOWANCE_MIB)
            };
            DeviceVramEstimate {
                node_id: device.node_id,
                layers: device.layers,
                estimated_vram_mib,
                available_vram_mib: device.available_vram_mib,
                fits: estimated_vram_mib <= device.available_vram_mib,
            }
        })
        .collect();
    let cpu_layers = request.total_layers - gpu_layers;
    Ok(SplitEstimate {
        total_layers: request.total_layers,
        gpu_layers,
        cpu_layers,
        estimated_cpu_ram_mib: proportional_mib(
            request.model_mib,
            cpu_layers,
            request.total_layers,
        ),
        uses_attention_metadata,
        devices,
    })
}

fn proportional_mib(total_mib: u64, part: u32, total: u32) -> u64 {
    (total_mib as u128 * part as u128).div_ceil(total as u128) as u64
}

fn kv_cache_mib_per_layer(request: &SplitEstimateRequest) -> u64 {
    let Some(embedding_length) = request.embedding_length else {
        return fallback_kv_mib_per_layer(request.context_size);
    };
    let Some(head_count) = request.attention_head_count.filter(|value| *value > 0) else {
        return fallback_kv_mib_per_layer(request.context_size);
    };
    let Some(kv_heads) = request.attention_head_count_kv.filter(|value| *value > 0) else {
        return fallback_kv_mib_per_layer(request.context_size);
    };
    let head_width = embedding_length.div_ceil(head_count) as u128;
    let bytes = request.context_size as u128
        * head_width
        * kv_heads as u128
        * 2 // K and V
        * 2; // F16 bytes
    bytes.div_ceil(1_048_576) as u64
}

fn fallback_kv_mib_per_layer(context_size: u32) -> u64 {
    (context_size as u64).div_ceil(4096) * 16
}

pub fn recommend_topology(request: CapacityRequest) -> CapacityRecommendation {
    let usable: Vec<u64> = request
        .gpu_mib
        .iter()
        .map(|v| v.saturating_sub(GPU_MARGIN_MIB))
        .collect();
    let context_size = request
        .requested_context
        .unwrap_or(DEFAULT_CONTEXT)
        .max(MIN_CONTEXT);
    if let Some((gpu_index, _)) = usable
        .iter()
        .enumerate()
        .filter(|(_, v)| **v >= request.model_mib)
        .max_by_key(|(_, v)| **v)
    {
        return CapacityRecommendation {
            outcome: CapacityOutcome::SingleNode { gpu_index },
            context_size,
        };
    }
    let gpu_total: u64 = usable.iter().sum();
    let splits = if gpu_total == 0 {
        vec![]
    } else {
        usable
            .iter()
            .map(|v| *v as f64 / gpu_total as f64)
            .collect()
    };
    if gpu_total >= request.model_mib {
        return CapacityRecommendation {
            outcome: CapacityOutcome::Distributed { splits },
            context_size,
        };
    }
    let reserve = MIN_RAM_RESERVE_MIB.max(request.system_ram_mib / 10);
    let usable_ram = request.system_ram_mib.saturating_sub(reserve);
    let spill_mib = request.model_mib.saturating_sub(gpu_total);
    let outcome = if spill_mib <= usable_ram {
        CapacityOutcome::RamSpill { splits, spill_mib }
    } else {
        CapacityOutcome::Insufficient
    };
    CapacityRecommendation {
        outcome,
        context_size,
    }
}
