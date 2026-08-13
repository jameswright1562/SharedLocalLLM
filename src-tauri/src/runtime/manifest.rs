use std::path::PathBuf;

use serde::Deserialize;

use crate::types::{ErrorPayload, RuntimeStatus};

const MANIFEST_JSON: &str = include_str!("../../../public/runtime/llama-cpp-manifest.json");

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeManifest {
    pub enabled: bool,
    pub release: Option<RuntimeRelease>,
    pub required_executables: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeRelease {
    pub tag: String,
    pub llama_cpp: RuntimeAsset,
    pub cuda_runtime: RuntimeAsset,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeAsset {
    pub url: String,
    pub size: u64,
    pub sha256: String,
}

pub fn manifest() -> Result<RuntimeManifest, ErrorPayload> {
    serde_json::from_str(MANIFEST_JSON)
        .map_err(|error| ErrorPayload::new("runtime_manifest_invalid", error.to_string(), None))
}

pub fn runtime_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("SharedLocalLLM")
        .join("runtime")
}

pub fn status() -> RuntimeStatus {
    let Ok(manifest) = manifest() else {
        return RuntimeStatus {
            status: "error".into(),
            version: None,
            error: Some("The bundled runtime manifest is invalid.".into()),
        };
    };
    let Some(release) = manifest.release else {
        return RuntimeStatus {
            status: "missing".into(),
            version: None,
            error: Some("No pinned llama.cpp release is configured.".into()),
        };
    };
    let current = runtime_root().join("current");
    let missing: Vec<_> = manifest
        .required_executables
        .iter()
        .filter(|name| !current.join(name).is_file())
        .collect();
    if missing.is_empty() {
        RuntimeStatus {
            status: "ready".into(),
            version: Some(format!("llama.cpp {}", release.tag)),
            error: None,
        }
    } else {
        RuntimeStatus {
            status: "missing".into(),
            version: Some(format!("llama.cpp {}", release.tag)),
            error: Some(format!(
                "Missing runtime files: {}",
                missing
                    .into_iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}
