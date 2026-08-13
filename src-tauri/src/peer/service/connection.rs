use std::{
    collections::HashMap,
    net::SocketAddr,
    process::Command,
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
use super::{PeerPairingEvent, PeerServerConfig};
use crate::{
    peer::{
        channel::{receive_encrypted, send_encrypted},
        crypto,
        protocol::{self, ClientHello, Request, Response},
        tunnel::serve_rpc,
    },
    types::ErrorPayload,
};

pub(super) struct ServerState {
    device_id: String,
    device_name: String,
    capabilities: Value,
    pub(super) pairing_code: Mutex<Option<String>>,
    pub(super) pairing_expires_at: Instant,
    pairing_completed: watch::Sender<Option<PeerPairingEvent>>,
    pending_pairing: Mutex<Option<PeerPairingEvent>>,
    pub(super) trusted: Mutex<HashMap<String, String>>,
    rpc_target: SocketAddr,
    rpc_command: Option<Vec<String>>,
    worker: Mutex<Option<ManagedChild>>,
}

impl ServerState {
    pub(super) fn new(
        config: PeerServerConfig,
        pairing_completed: watch::Sender<Option<PeerPairingEvent>>,
    ) -> Self {
        Self {
            device_id: config.device_id,
            device_name: config.device_name,
            capabilities: config.capabilities,
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
            rpc_target: config.rpc_target,
            rpc_command: config.rpc_command,
            worker: Mutex::new(None),
        }
    }
}

struct ManagedChild(std::process::Child);
impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(super) async fn handle_connection(
    mut socket: TcpStream,
    state: Arc<ServerState>,
) -> Result<(), ErrorPayload> {
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
        Request::Benchmark { payload } => {
            ensure_trusted(&state, &hello.device_id).await?;
            send_encrypted(&mut socket, &mut noise, &Response::Benchmark { payload }).await
        }
        Request::RpcTunnel => rpc_tunnel(socket, noise, state, &hello.device_id).await,
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
    state
        .trusted
        .lock()
        .await
        .insert(attempt.device_id.clone(), channel_key.clone());
    *state.pairing_code.lock().await = None;
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
            value: state.capabilities.clone(),
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

async fn rpc_tunnel(
    socket: TcpStream,
    mut noise: snow::TransportState,
    state: Arc<ServerState>,
    device_id: &str,
) -> Result<(), ErrorPayload> {
    ensure_trusted(&state, device_id).await?;
    ensure_worker(&state).await?;
    let rpc = connect_rpc(state.rpc_target).await?;
    let mut socket = socket;
    send_encrypted(&mut socket, &mut noise, &Response::RpcReady).await?;
    serve_rpc(socket, noise, rpc).await
}

async fn ensure_worker(state: &ServerState) -> Result<(), ErrorPayload> {
    let mut worker = state.worker.lock().await;
    let running = match worker.as_mut() {
        Some(child) => child.0.try_wait().map_err(protocol::io_error)?.is_none(),
        None => false,
    };
    if running {
        return Ok(());
    }
    *worker = None;
    if let Some((program, args)) = state
        .rpc_command
        .as_ref()
        .and_then(|command| command.split_first())
    {
        *worker = Some(ManagedChild(
            Command::new(program)
                .args(args)
                .spawn()
                .map_err(protocol::io_error)?,
        ));
    }
    Ok(())
}

async fn connect_rpc(target: SocketAddr) -> Result<TcpStream, ErrorPayload> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        match TcpStream::connect(target).await {
            Ok(stream) => return Ok(stream),
            Err(_) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            Err(error) => return Err(protocol::io_error(error)),
        }
    }
}
