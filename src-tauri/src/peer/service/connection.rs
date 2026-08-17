use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use serde_json::Value;
use tokio::{net::TcpStream, sync::Mutex};

use super::worker::{self, ManagedChild};
use super::{PeerConnectedEvent, PeerServerConfig};
use crate::{
    peer::protocol::{self, ClientHello, Request, Response},
    types::ErrorPayload,
};

pub(super) struct ServerState {
    device_id: String,
    device_name: String,
    pub(super) capabilities: Mutex<Value>,
    connected: std::sync::Mutex<Vec<PeerConnectedEvent>>,
    pub(super) rpc_target: Mutex<Option<SocketAddr>>,
    pub(super) worker: Mutex<Option<ManagedChild>>,
    pub(super) catalogue: Mutex<Value>,
    pub(super) api_key: Mutex<String>,
    pub(super) api_port: Mutex<u16>,
}

impl ServerState {
    pub(super) fn new(config: PeerServerConfig) -> Self {
        Self {
            device_id: config.device_id,
            device_name: config.device_name,
            capabilities: Mutex::new(config.capabilities),
            connected: std::sync::Mutex::new(Vec::new()),
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

    pub(super) fn take_connected(&self) -> Vec<PeerConnectedEvent> {
        std::mem::take(&mut *self.connected.lock().unwrap())
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
    eprintln!(
        "{} INFO peer_hello: device={} from={}",
        crate::pairing::now(),
        hello.device_id,
        source
    );
    let request: Request = protocol::read_plain(&mut socket).await?;
    eprintln!(
        "{} INFO peer_request: device={} request={} from={}",
        crate::pairing::now(),
        hello.device_id,
        request_kind(&request),
        source
    );
    let result = match request {
        Request::Connect {
            version,
            device_id,
            device_name,
            capabilities,
        } => {
            connect(
                &mut socket,
                &state,
                version,
                device_id,
                device_name,
                capabilities,
                source,
            )
            .await
        }
        Request::Heartbeat { version, .. } => {
            protocol::check_version(version)?;
            protocol::write_plain(&mut socket, &Response::Heartbeat).await
        }
        Request::Capabilities => capabilities(&mut socket, &state).await,
        Request::Benchmark { size } => worker::handle_benchmark(&mut socket, size).await,
        Request::RpcTunnel => worker::rpc_tunnel(socket, state).await,
        Request::StopWorker => worker::handle_stop_worker(&mut socket, &state).await,
        Request::Models => worker::handle_models(&mut socket, &state).await,
        request @ Request::ProxyChat { .. } => {
            worker::handle_proxy_chat(&mut socket, &state, request).await
        }
    };
    if let Err(error) = &result {
        eprintln!(
            "{} WARN peer_request_failed: device={} error={}",
            crate::pairing::now(),
            hello.device_id,
            error
        );
    }
    result
}

fn request_kind(request: &Request) -> &'static str {
    match request {
        Request::Connect { .. } => "connect",
        Request::Heartbeat { .. } => "heartbeat",
        Request::Capabilities => "capabilities",
        Request::Benchmark { .. } => "benchmark",
        Request::RpcTunnel => "rpc_tunnel",
        Request::StopWorker => "stop_worker",
        Request::Models => "models",
        Request::ProxyChat { .. } => "proxy_chat",
    }
}

#[allow(clippy::too_many_arguments)]
async fn connect(
    socket: &mut TcpStream,
    state: &ServerState,
    version: u16,
    device_id: String,
    device_name: String,
    capabilities: Value,
    source: SocketAddr,
) -> Result<(), ErrorPayload> {
    protocol::check_version(version)?;
    state.connected.lock().unwrap().push(PeerConnectedEvent {
        device_id: device_id.clone(),
        device_name,
        capabilities,
        source,
    });
    protocol::write_plain(
        socket,
        &Response::Connected {
            device_id: state.device_id.clone(),
            device_name: state.device_name.clone(),
        },
    )
    .await
}

async fn capabilities(socket: &mut TcpStream, state: &ServerState) -> Result<(), ErrorPayload> {
    protocol::write_plain(
        socket,
        &Response::Capabilities {
            value: state.capabilities.lock().await.clone(),
        },
    )
    .await
}
