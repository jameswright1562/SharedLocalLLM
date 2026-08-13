use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use rand::{distributions::Alphanumeric, Rng};
use serde_json::Value;
use tokio::{
    net::TcpStream,
    sync::{watch, Mutex},
};

use super::auth::{credential_for, ensure_trusted};
use super::worker::{self, ManagedChild};
use super::{PeerPairingEvent, PeerServerConfig};
use crate::{
    peer::{
        channel::{receive_encrypted, send_encrypted},
        crypto,
        protocol::{self, ClientHello, Request, Response},
    },
    types::ErrorPayload,
};

pub(super) struct ServerState {
    device_id: String,
    device_name: String,
    pub(super) capabilities: Mutex<Value>,
    pub(super) pairing_code: Mutex<Option<String>>,
    pub(super) pairing_expires_at: Instant,
    pairing_completed: watch::Sender<Option<PeerPairingEvent>>,
    pending_pairing: Mutex<Option<PeerPairingEvent>>,
    pub(super) trusted: Mutex<HashMap<String, String>>,
    pub(super) rpc_binary: Option<PathBuf>,
    pub(super) rpc_override: Option<SocketAddr>,
    pub(super) rpc_target: Mutex<Option<SocketAddr>>,
    pub(super) worker: Mutex<Option<ManagedChild>>,
    pub(super) catalogue: Mutex<Value>,
    pub(super) api_key: Mutex<String>,
    pub(super) api_port: Mutex<u16>,
}

impl ServerState {
    pub(super) fn new(
        config: PeerServerConfig,
        pairing_completed: watch::Sender<Option<PeerPairingEvent>>,
    ) -> Self {
        Self {
            device_id: config.device_id,
            device_name: config.device_name,
            capabilities: Mutex::new(config.capabilities),
            pairing_code: Mutex::new(config.pairing_code),
            pairing_expires_at: Instant::now() + Duration::from_secs(300),
            pairing_completed,
            pending_pairing: Mutex::new(None),
            trusted: Mutex::new(
                config
                    .trusted_peers
                    .into_iter()
                    .map(|peer| (peer.device_id, peer.channel_key))
                    .collect(),
            ),
            rpc_binary: config.rpc_binary,
            rpc_override: config.rpc_override,
            rpc_target: Mutex::new(config.rpc_override),
            worker: Mutex::new(None),
            catalogue: Mutex::new(config.catalogue),
            api_key: Mutex::new(config.api_key),
            api_port: Mutex::new(config.api_port),
        }
    }

    pub(super) async fn set_capabilities(&self, value: Value) {
        *self.capabilities.lock().await = value;
    }

    pub(super) async fn set_catalogue(&self, value: Value) {
        *self.catalogue.lock().await = value;
    }

    pub(super) async fn set_api(&self, api_key: String, api_port: u16) {
        *self.api_key.lock().await = api_key;
        *self.api_port.lock().await = api_port;
    }

    pub(super) async fn stop_local_worker(&self) {
        worker::stop_worker(self).await;
    }
}

pub(super) async fn handle_connection(
    mut socket: TcpStream,
    state: Arc<ServerState>,
) -> Result<(), ErrorPayload> {
    let _ = socket.set_nodelay(true);
    let hello: ClientHello =
        tokio::time::timeout(Duration::from_secs(5), protocol::read_plain(&mut socket))
            .await
            .map_err(|_| {
                ErrorPayload::new("peer_timeout", "The peer handshake timed out.", None)
            })??;
    let source = socket.peer_addr().map_err(protocol::io_error)?;
    let credential = credential_for(&state, &hello).await?;
    let mut handshake = crypto::responder(&credential)?;
    let incoming: Vec<u8> = protocol::read_plain(&mut socket).await?;
    handshake
        .read_message(&incoming, &mut [])
        .map_err(crypto::noise_error)?;
    let mut outgoing = [0_u8; 1024];
    let count = handshake
        .write_message(&[], &mut outgoing)
        .map_err(crypto::noise_error)?;
    protocol::write_plain(&mut socket, &outgoing[..count].to_vec()).await?;
    let mut noise = crypto::transport(handshake)?;
    let request: Request = receive_encrypted(&mut socket, &mut noise).await?;
    match request {
        Request::Pair {
            version,
            device_id,
            device_name,
            capabilities,
        } => {
            pair(
                &mut socket,
                &mut noise,
                state,
                PairAttempt {
                    hello,
                    version,
                    device_id,
                    device_name,
                    capabilities,
                    source,
                },
            )
            .await
        }
        Request::Heartbeat { version, device_id } => {
            protocol::check_version(version)?;
            ensure_trusted(&state, &device_id).await?;
            send_encrypted(&mut socket, &mut noise, &Response::Heartbeat).await
        }
        Request::Capabilities => capabilities(&mut socket, &mut noise, &state, &hello).await,
        Request::Benchmark { size } => {
            worker::handle_benchmark(&mut socket, &mut noise, &state, &hello.device_id, size).await
        }
        Request::RpcTunnel => worker::rpc_tunnel(socket, noise, state, &hello.device_id).await,
        Request::StopWorker => {
            worker::handle_stop_worker(&mut socket, &mut noise, &state, &hello.device_id).await
        }
        Request::Models => {
            worker::handle_models(&mut socket, &mut noise, &state, &hello.device_id).await
        }
        request @ Request::ProxyChat { .. } => {
            worker::handle_proxy_chat(&mut socket, &mut noise, &state, &hello.device_id, request)
                .await
        }
    }
}

struct PairAttempt {
    hello: ClientHello,
    version: u16,
    device_id: String,
    device_name: String,
    capabilities: Value,
    source: SocketAddr,
}

async fn pair(
    socket: &mut TcpStream,
    noise: &mut snow::TransportState,
    state: Arc<ServerState>,
    attempt: PairAttempt,
) -> Result<(), ErrorPayload> {
    protocol::check_version(attempt.version)?;
    if !attempt.hello.pairing || attempt.hello.device_id != attempt.device_id {
        return Err(ErrorPayload::new(
            "pairing_identity_mismatch",
            "The pairing identity changed during authentication.",
            None,
        ));
    }
    let channel_key: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(64)
        .map(char::from)
        .collect();
    send_encrypted(
        socket,
        noise,
        &Response::Paired {
            device_id: state.device_id.clone(),
            device_name: state.device_name.clone(),
            channel_key: channel_key.clone(),
        },
    )
    .await?;
    state
        .trusted
        .lock()
        .await
        .insert(attempt.device_id.clone(), channel_key.clone());
    *state.pairing_code.lock().await = None;
    *state.pending_pairing.lock().await = Some(PeerPairingEvent {
        device_id: attempt.device_id,
        device_name: attempt.device_name,
        channel_key,
        capabilities: attempt.capabilities,
        source: attempt.source,
    });
    Ok(())
}

async fn capabilities(
    socket: &mut TcpStream,
    noise: &mut snow::TransportState,
    state: &ServerState,
    hello: &ClientHello,
) -> Result<(), ErrorPayload> {
    ensure_trusted(state, &hello.device_id).await?;
    send_encrypted(
        socket,
        noise,
        &Response::Capabilities {
            value: state.capabilities.lock().await.clone(),
        },
    )
    .await?;
    let mut pending = state.pending_pairing.lock().await;
    if pending
        .as_ref()
        .is_some_and(|event| event.device_id == hello.device_id)
    {
        let _ = state.pairing_completed.send(pending.take());
    }
    Ok(())
}
