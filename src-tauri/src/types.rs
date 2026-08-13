use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_gb: f64,
    pub vram_available_gb: f64,
    pub driver_version: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NetworkAdapterInfo {
    pub name: String,
    pub kind: String,
    pub link_speed_mbps: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NodeCapabilities {
    pub id: String,
    pub name: String,
    pub online: bool,
    pub role: String,
    pub cpu: String,
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,
    pub gpu: GpuInfo,
    pub adapter: NetworkAdapterInfo,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelLocation {
    pub node_id: String,
    pub path: String,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelDirectory {
    pub id: String,
    pub node_id: String,
    pub path: String,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelRecord {
    pub id: String,
    pub name: String,
    pub architecture: String,
    pub quantization: String,
    pub size_bytes: u64,
    pub context_length: u32,
    pub layer_count: Option<u32>,
    pub embedding_length: Option<u32>,
    pub attention_head_count: Option<u32>,
    pub attention_head_count_kv: Option<u32>,
    pub capability: String,
    pub shards: usize,
    pub locations: Vec<ModelLocation>,
    pub fit: String,
    #[serde(skip_serializing)]
    pub shard_paths: Vec<String>,
    #[serde(skip_serializing)]
    pub projector: Option<String>,
    #[serde(skip_serializing)]
    pub vision_capable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GpuLayerAllocation {
    pub node_id: String,
    pub layers: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadConfig {
    pub context_size: u32,
    pub gpu_layers: Vec<GpuLayerAllocation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NetworkBenchmark {
    pub down_mbps: f64,
    pub up_mbps: f64,
    pub latency_median_ms: f64,
    pub latency_p95_ms: f64,
    pub jitter_ms: f64,
    pub packet_loss_percent: f64,
    pub classification: String,
    pub adapter: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ClusterProfile {
    pub coordinator_node_id: String,
    pub worker_node_id: Option<String>,
    pub model_id: String,
    pub context_size: u32,
    pub split_mode: String,
    pub tensor_split: Vec<f64>,
    pub ram_spill_mib: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InferenceBenchmark {
    pub id: String,
    pub model_name: String,
    pub topology: String,
    pub prompt_tokens_per_second: f64,
    pub generation_tokens_per_second: f64,
    pub load_time_seconds: f64,
    pub memory_peak_gb: f64,
    pub recommended: bool,
    pub ran_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ClusterSession {
    pub status: String,
    pub coordinator_node_id: Option<String>,
    pub worker_node_id: Option<String>,
    pub model_id: Option<String>,
    pub error: Option<String>,
}

impl Default for ClusterSession {
    fn default() -> Self {
        Self {
            status: "idle".into(),
            coordinator_node_id: None,
            worker_node_id: None,
            model_id: None,
            error: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatus {
    pub status: String,
    pub version: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AppSnapshot {
    pub setup_complete: bool,
    pub runtime: RuntimeStatus,
    pub device_name: String,
    pub api_port: u16,
    pub autostart: bool,
    pub nodes: Vec<NodeCapabilities>,
    pub models: Vec<ModelRecord>,
    pub model_directories: Vec<ModelDirectory>,
    pub network: Option<NetworkBenchmark>,
    pub cluster: ClusterSession,
    pub benchmarks: Vec<InferenceBenchmark>,
    pub logs: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ApiConfig {
    pub url: String,
    pub api_key: String,
    pub healthy: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub image_names: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSettings {
    pub system_prompt: String,
    pub temperature: f64,
    pub max_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PairingCode {
    pub code: String,
    pub expires_in_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    pub device_name: String,
    pub api_port: u16,
    pub autostart: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorPayload {
    pub code: String,
    pub message: String,
    pub action: Option<String>,
}

impl ErrorPayload {
    pub fn new(
        code: impl Into<String>,
        message: impl Into<String>,
        action: impl Into<Option<String>>,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            action: action.into(),
        }
    }
}

impl std::fmt::Display for ErrorPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for ErrorPayload {}
