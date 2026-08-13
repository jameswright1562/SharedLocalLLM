mod auth;
mod connection;

use std::{net::SocketAddr, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    net::TcpListener,
    sync::{oneshot, watch},
    task::JoinHandle,
};

use super::protocol;
use crate::types::ErrorPayload;
use connection::{handle_connection, ServerState};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustedPeer {
    pub device_id: String,
    pub device_name: String,
    pub channel_key: String,
}

pub struct PeerServerConfig {
    pub bind: SocketAddr,
    pub device_id: String,
    pub device_name: String,
    pub capabilities: Value,
    pub pairing_code: Option<String>,
    pub trusted_peers: Vec<TrustedPeer>,
    pub rpc_target: SocketAddr,
    pub rpc_command: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
pub struct PeerPairingEvent {
    pub device_id: String,
    pub device_name: String,
    pub channel_key: String,
    pub capabilities: Value,
    pub source: SocketAddr,
}

pub struct PeerServer {
    address: SocketAddr,
    pairing_completed: watch::Receiver<Option<PeerPairingEvent>>,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl PeerServer {
    pub async fn start(config: PeerServerConfig) -> Result<Self, ErrorPayload> {
        if !config.rpc_target.ip().is_loopback() {
            return Err(ErrorPayload::new(
                "rpc_target_rejected",
                "The worker RPC target must be loopback.",
                None,
            ));
        }
        let listener = TcpListener::bind(config.bind)
            .await
            .map_err(protocol::io_error)?;
        let address = listener.local_addr().map_err(protocol::io_error)?;
        let (stop, mut stopped) = oneshot::channel();
        let (pairing_completed_tx, pairing_completed) = watch::channel(None);
        let config = Arc::new(ServerState::new(config, pairing_completed_tx));
        let connections = Arc::new(tokio::sync::Semaphore::new(32));
        let task = tokio::spawn(async move {
            let mut tasks = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    _ = &mut stopped => break,
                    accepted = listener.accept() => {
                        let Ok((socket, _)) = accepted else { break };
                        let config = config.clone();
                        let Ok(permit) = connections.clone().try_acquire_owned() else { continue };
                        tasks.spawn(async move { let _permit = permit; let _ = handle_connection(socket, config).await; });
                    }
                }
            }
            tasks.shutdown().await;
        });
        Ok(Self {
            address,
            pairing_completed,
            stop: Some(stop),
            task,
        })
    }
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    pub fn pairing_completion(&self) -> watch::Receiver<Option<PeerPairingEvent>> {
        self.pairing_completed.clone()
    }
    pub async fn shutdown(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let _ = self.task.await;
    }
}
