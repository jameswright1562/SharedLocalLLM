mod connection;
mod worker;

use std::{net::SocketAddr, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

use super::protocol;
use crate::types::ErrorPayload;
use connection::{handle_connection, ServerState};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerConnectedEvent {
    pub device_id: String,
    pub device_name: String,
    pub capabilities: Value,
    pub source: SocketAddr,
}

pub struct PeerServerConfig {
    pub bind: SocketAddr,
    pub device_id: String,
    pub device_name: String,
    pub capabilities: Value,
    pub rpc_target: SocketAddr,
    pub catalogue: Value,
    pub api_key: String,
    pub api_port: u16,
}

pub struct PeerServer {
    address: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
    state: Arc<connection::ServerState>,
}

impl PeerServer {
    pub async fn start(config: PeerServerConfig) -> Result<Self, ErrorPayload> {
        let listener = TcpListener::bind(config.bind)
            .await
            .map_err(protocol::io_error)?;
        let address = listener.local_addr().map_err(protocol::io_error)?;
        let (stop, mut stopped) = oneshot::channel();
        let state = Arc::new(ServerState::new(config));
        let connections = Arc::new(tokio::sync::Semaphore::new(32));
        let accept_state = state.clone();
        let task = tokio::spawn(async move {
            let mut tasks = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    _ = &mut stopped => break,
                    accepted = listener.accept() => {
                        let Ok((socket, _)) = accepted else { break };
                        let source = socket
                            .peer_addr()
                            .map(|address| address.to_string())
                            .unwrap_or_else(|_| "unknown".into());
                        eprintln!(
                            "{} INFO peer_connection: accepted from {}",
                            crate::pairing::now(),
                            source
                        );
                        let accept_state = accept_state.clone();
                        let Ok(permit) = connections.clone().try_acquire_owned() else { continue };
                        tasks.spawn(async move { let _permit = permit; let _ = handle_connection(socket, accept_state).await; });
                    }
                }
            }
            tasks.shutdown().await;
        });
        Ok(Self {
            address,
            stop: Some(stop),
            task,
            state,
        })
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    pub async fn set_capabilities(&self, value: Value) {
        self.state.set_capabilities(value).await;
    }

    pub async fn set_catalogue(&self, value: Value) {
        self.state.set_catalogue(value).await;
    }

    pub async fn set_api(&self, api_key: String, api_port: u16) {
        self.state.set_api(api_key, api_port).await;
    }

    pub async fn stop_local_worker(&self) {
        self.state.stop_local_worker().await;
    }

    pub fn take_connected(&self) -> Vec<PeerConnectedEvent> {
        self.state.take_connected()
    }

    pub async fn shutdown(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let _ = self.task.await;
    }
}
