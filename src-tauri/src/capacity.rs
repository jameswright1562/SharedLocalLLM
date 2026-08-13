use serde::{Deserialize, Serialize};

pub const GPU_MARGIN_MIB: u64 = 1024;
pub const DEFAULT_CONTEXT: u32 = 8192;
pub const MIN_CONTEXT: u32 = 4096;
pub const MIN_RAM_RESERVE_MIB: u64 = 6 * 1024;

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
