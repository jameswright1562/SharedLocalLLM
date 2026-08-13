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

use super::PeerServerConfig;
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
    pairing_code: Mutex<Option<String>>,
    pairing_expires_at: Instant,
    pairing_completed: watch::Sender<bool>,
    pairing_completed_device: Mutex<Option<String>>,
    trusted: Mutex<HashMap<String, String>>,
    rpc_target: SocketAddr,
    rpc_command: Option<Vec<String>>,
    worker: Mutex<Option<ManagedChild>>,
}

impl ServerState {
    pub(super) fn new(config: PeerServerConfig, pairing_completed: watch::Sender<bool>) -> Self {
        Self {
            device_id: config.device_id,
            device_name: config.device_name,
            capabilities: config.capabilities,
            pairing_code: Mutex::new(config.pairing_code),
            pairing_expires_at: Instant::now() + Duration::from_secs(300),
            pairing_completed,
            pairing_completed_device: Mutex::new(None),
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
            device_name: _,
        } => pair(&mut socket, &mut noise, state, hello, version, device_id).await,
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

async fn credential_for(state: &ServerState, hello: &ClientHello) -> Result<String, ErrorPayload> {
    if hello.pairing {
        if Instant::now() > state.pairing_expires_at {
            return Err(ErrorPayload::new(
                "pairing_code_expired",
                "The pairing code expired.",
                Some("Generate a new code.".into()),
            ));
        }
        state.pairing_code.lock().await.clone().ok_or_else(|| {
            ErrorPayload::new(
                "pairing_unavailable",
                "This peer is not currently showing a pairing code.",
                None,
            )
        })
    } else {
        state
            .trusted
            .lock()
            .await
            .get(&hello.device_id)
            .cloned()
            .ok_or_else(untrusted_error)
    }
}

async fn pair(
    socket: &mut TcpStream,
    noise: &mut snow::TransportState,
    state: Arc<ServerState>,
    hello: ClientHello,
    version: u16,
    device_id: String,
) -> Result<(), ErrorPayload> {
    protocol::check_version(version)?;
    if !hello.pairing || hello.device_id != device_id {
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
        .insert(device_id.clone(), channel_key.clone());
    *state.pairing_code.lock().await = None;
    send_encrypted(
        socket,
        noise,
        &Response::Paired {
            device_id: state.device_id.clone(),
            device_name: state.device_name.clone(),
            channel_key,
        },
    )
    .await?;
    *state.pairing_completed_device.lock().await = Some(device_id);
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
    let mut completed_device = state.pairing_completed_device.lock().await;
    if completed_device.as_deref() == Some(hello.device_id.as_str()) {
        completed_device.take();
        let _ = state.pairing_completed.send(true);
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

async fn ensure_trusted(state: &ServerState, id: &str) -> Result<(), ErrorPayload> {
    if state.trusted.lock().await.contains_key(id) {
        Ok(())
    } else {
        Err(untrusted_error())
    }
}

fn untrusted_error() -> ErrorPayload {
    ErrorPayload::new(
        "peer_untrusted",
        "The peer identity is not trusted.",
        Some("Pair the computers again.".into()),
    )
}
