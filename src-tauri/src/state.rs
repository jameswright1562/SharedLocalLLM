mod persistence;
mod placement;

use std::{
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

pub use persistence::{data_root, directory_for, logs_root, peer_secret_path, regenerate_key};
use persistence::{new_api_key, read_settings, save_settings, secrets_path, PersistedSettings};

use crate::{
    hardware,
    models::{default_lm_studio_root, discover_gguf_models, lms_catalog_roots},
    pairing::{PairingManager, PeerRecord},
    peer::{DiscoveryBroadcaster, PeerClient, PeerServer, RpcForwarder},
    runtime::{self, ProcessManager},
    secrets,
    types::*,
};

pub struct AppState {
    pub inner: Mutex<InnerState>,
    pub pairing: Mutex<PairingManager>,
    pub processes: Mutex<ProcessManager>,
    pub peer: tokio::sync::Mutex<PeerRuntime>,
    pub chat_cancel: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    pub benchmark_cancel: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

#[derive(Default)]
pub struct PeerRuntime {
    pub server: Option<PeerServer>,
    pub discovery: Option<DiscoveryBroadcaster>,
    pub pairing_session_id: Option<String>,
    pub public_firewall_lease: Option<PathBuf>,
    pub client: Option<std::sync::Arc<PeerClient>>,
    pub forwarder: Option<RpcForwarder>,
}

pub struct InnerState {
    pub local: NodeCapabilities,
    pub peers: Vec<PeerRecord>,
    pub directories: Vec<ModelDirectory>,
    pub models: Vec<ModelRecord>,
    pub network: Option<NetworkBenchmark>,
    pub cluster: ClusterSession,
    pub benchmarks: Vec<InferenceBenchmark>,
    pub logs: Vec<String>,
    pub api_key: String,
    pub setup_complete: bool,
    pub api_port: u16,
    pub autostart: bool,
}

impl AppState {
    pub fn new() -> Self {
        let persisted = read_settings();
        let mut local = hardware::probe_local();
        if let Some(device_name) = &persisted.device_name {
            local.name = device_name.clone();
        }
        let mut directories = Vec::new();
        if let Some(path) = default_lm_studio_root() {
            directories.push(directory_for(&path, "lm-studio", &local.id));
        }
        for path in persisted.custom_model_directories.iter().map(PathBuf::from) {
            directories.push(directory_for(&path, "custom", &local.id));
        }
        let (api_key, secret_log) = match secrets::load(&secrets_path()) {
            Ok(Some(value)) => match String::from_utf8(value) {
                Ok(key) => (key, None),
                Err(error) => (
                    new_api_key(),
                    Some(format!("ERROR Protected API key is invalid UTF-8: {error}")),
                ),
            },
            Ok(None) => {
                let key = new_api_key();
                let error = secrets::store(&secrets_path(), key.as_bytes())
                    .err()
                    .map(|error| format!("ERROR Could not protect the generated API key: {error}"));
                (key, error)
            }
            Err(error) => (
                new_api_key(),
                Some(format!(
                    "ERROR Could not read the protected API key: {error}"
                )),
            ),
        };
        let mut logs = vec!["READY Backend initialized; raw RPC will bind loopback only".into()];
        if let Some(error) = secret_log {
            logs.push(error);
        }
        let state = Self {
            inner: Mutex::new(InnerState {
                local,
                peers: persisted.peers,
                directories,
                models: vec![],
                network: None,
                cluster: ClusterSession::default(),
                benchmarks: persisted.benchmarks,
                logs,
                api_key,
                setup_complete: persisted.setup_complete,
                api_port: persisted.api_port.unwrap_or(11435),
                autostart: persisted.autostart,
            }),
            pairing: Mutex::new(PairingManager::default()),
            processes: Mutex::new(ProcessManager::default()),
            peer: tokio::sync::Mutex::new(PeerRuntime::default()),
            chat_cancel: Mutex::new(None),
            benchmark_cancel: Mutex::new(None),
        };
        let _ = state.refresh_models_shared();
        state
    }

    pub fn lock(&self) -> Result<MutexGuard<'_, InnerState>, ErrorPayload> {
        self.inner.lock().map_err(|_| {
            ErrorPayload::new(
                "state_poisoned",
                "Application state could not be accessed.",
                Some("Restart SharedLocalLLM.".into()),
            )
        })
    }

    pub fn refresh_models_shared(&self) -> Result<Vec<ModelRecord>, ErrorPayload> {
        let (mut roots, local) = {
            let inner = self.lock()?;
            (
                inner
                    .directories
                    .iter()
                    .map(|d| PathBuf::from(&d.path))
                    .collect::<Vec<_>>(),
                inner.local.clone(),
            )
        };
        roots.extend(lms_catalog_roots());
        let mut models = discover_gguf_models(&roots)?;
        let peers = self.lock()?.peers.clone();
        placement::apply_fit(&mut models, &local, &peers);
        self.lock()?.models = models.clone();
        Ok(models)
    }

    pub fn snapshot(&self) -> Result<AppSnapshot, ErrorPayload> {
        let inner = self.lock()?;
        let mut nodes = vec![inner.local.clone()];
        nodes.extend(inner.peers.iter().map(|peer| {
            peer.capabilities
                .clone()
                .unwrap_or_else(|| NodeCapabilities {
                    id: peer.id.clone(),
                    name: peer.name.clone(),
                    online: false,
                    role: "available".into(),
                    ..NodeCapabilities::default()
                })
        }));
        Ok(AppSnapshot {
            setup_complete: inner.setup_complete,
            runtime: runtime::status(),
            device_name: inner.local.name.clone(),
            api_port: inner.api_port,
            autostart: inner.autostart,
            nodes,
            models: inner.models.clone(),
            model_directories: inner.directories.clone(),
            network: inner.network.clone(),
            cluster: inner.cluster.clone(),
            benchmarks: inner.benchmarks.clone(),
            logs: inner.logs.clone(),
        })
    }

    pub fn persist(&self) -> Result<(), ErrorPayload> {
        let inner = self.lock()?;
        let settings = PersistedSettings {
            custom_model_directories: inner
                .directories
                .iter()
                .filter(|d| d.source == "custom")
                .map(|d| d.path.clone())
                .collect(),
            peers: inner.peers.clone(),
            device_name: Some(inner.local.name.clone()),
            setup_complete: inner.setup_complete,
            api_port: Some(inner.api_port),
            autostart: inner.autostart,
            benchmarks: inner.benchmarks.clone(),
        };
        save_settings(&settings)
    }

    pub async fn peer_client(&self) -> Result<std::sync::Arc<PeerClient>, ErrorPayload> {
        if let Some(client) = self.peer.lock().await.client.clone() {
            return Ok(client);
        }
        let (local_id, peer) = {
            let inner = self.lock()?;
            (inner.local.id.clone(), inner.peers.first().cloned())
        };
        let peer = peer.ok_or_else(|| {
            ErrorPayload::new("peer_unavailable", "Pair another computer first.", None)
        })?;
        let endpoint = peer
            .address
            .as_deref()
            .ok_or_else(|| {
                ErrorPayload::new(
                    "peer_endpoint_missing",
                    "The paired computer has no saved endpoint.",
                    Some("Pair the computers again.".into()),
                )
            })?
            .parse()
            .map_err(|_| {
                ErrorPayload::new(
                    "peer_endpoint_invalid",
                    "The saved peer endpoint is invalid.",
                    Some("Pair the computers again.".into()),
                )
            })?;
        let channel_key = secrets::load(&peer_secret_path(&peer.id))?
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .ok_or_else(|| {
                ErrorPayload::new(
                    "peer_secret_missing",
                    "The protected peer credential is unavailable.",
                    Some("Pair the computers again.".into()),
                )
            })?;
        let client = std::sync::Arc::new(PeerClient::trusted(endpoint, channel_key, local_id));
        self.peer.lock().await.client = Some(client.clone());
        Ok(client)
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
