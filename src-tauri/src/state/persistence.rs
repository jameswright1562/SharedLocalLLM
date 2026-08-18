use std::{
    fs,
    path::{Path, PathBuf},
};

use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::{
    pairing::PeerRecord,
    types::{ErrorPayload, InferenceBenchmark, ModelDirectory},
};

#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(super) struct PersistedSettings {
    pub install_id: Option<String>,
    pub custom_model_directories: Vec<String>,
    pub peers: Vec<PeerRecord>,
    pub device_name: Option<String>,
    pub setup_complete: bool,
    pub api_port: Option<u16>,
    pub autostart: bool,
    pub benchmarks: Vec<InferenceBenchmark>,
}

pub(super) fn resolve_install_id(saved: Option<&str>) -> String {
    saved
        .filter(|id| id.starts_with("device-") && id.len() > "device-".len())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("device-{}", uuid::Uuid::new_v4()))
}

pub(super) fn read_settings() -> PersistedSettings {
    fs::read(settings_path())
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub(super) fn save_settings(settings: &PersistedSettings) -> Result<(), ErrorPayload> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(state_io)?;
    }
    let data = serde_json::to_vec_pretty(settings)
        .map_err(|error| ErrorPayload::new("settings_encode", error.to_string(), None))?;
    fs::write(path, data).map_err(state_io)
}

pub fn directory_for(path: &Path, source: &str, node_id: &str) -> ModelDirectory {
    let mut digest = sha2::Sha256::new();
    digest.update(path.to_string_lossy().as_bytes());
    ModelDirectory {
        id: hex::encode(digest.finalize())[..12].into(),
        node_id: node_id.into(),
        path: path.to_string_lossy().into_owned(),
        source: source.into(),
    }
}

pub fn data_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("SharedLocalLLM")
}
pub fn logs_root() -> PathBuf {
    data_root().join("logs")
}
fn settings_path() -> PathBuf {
    data_root().join("settings.json")
}
pub(super) fn secrets_path() -> PathBuf {
    data_root().join("secrets.dat")
}

pub(super) fn new_api_key() -> String {
    format!(
        "sk-local-{}",
        rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect::<String>()
    )
}
pub fn regenerate_key() -> String {
    new_api_key()
}

fn state_io(error: std::io::Error) -> ErrorPayload {
    ErrorPayload::new(
        "settings_io",
        error.to_string(),
        Some("Check the application data folder permissions.".into()),
    )
}

#[cfg(test)]
mod tests {
    use super::resolve_install_id;

    #[test]
    fn install_identity_is_stable_and_legacy_placeholder_is_replaced() {
        assert_eq!(
            resolve_install_id(Some("device-existing")),
            "device-existing"
        );
        let replacement = resolve_install_id(Some("local-node"));
        assert!(replacement.starts_with("device-"));
        assert_ne!(replacement, "local-node");
    }
}
