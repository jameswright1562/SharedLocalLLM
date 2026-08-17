use std::{
    net::{Ipv4Addr, SocketAddr},
    process::Command,
    sync::Arc,
    time::Duration,
};

use tokio::net::{TcpListener, TcpStream};

use super::connection::ServerState;
use crate::{
    peer::{
        protocol::{self, Request, Response},
        tunnel::bridge,
    },
    runtime::ProcessJob,
    types::ErrorPayload,
};

pub(super) struct ManagedChild {
    child: std::process::Child,
    _job: Option<ProcessJob>,
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(super) async fn rpc_tunnel(
    mut socket: TcpStream,
    state: Arc<ServerState>,
) -> Result<(), ErrorPayload> {
    let target = ensure_worker(&state).await?;
    let rpc = connect_rpc(target).await?;
    let _ = socket.set_nodelay(true);
    let _ = rpc.set_nodelay(true);
    protocol::write_plain(&mut socket, &Response::RpcReady).await?;
    bridge(socket, rpc).await
}

pub(super) async fn stop_worker(state: &ServerState) {
    *state.worker.lock().await = None;
    *state.rpc_target.lock().await = None;
}

pub(super) async fn handle_stop_worker(
    socket: &mut TcpStream,
    state: &ServerState,
) -> Result<(), ErrorPayload> {
    stop_worker(state).await;
    protocol::write_plain(socket, &Response::WorkerStopped).await
}

pub(super) async fn handle_models(
    socket: &mut TcpStream,
    state: &ServerState,
) -> Result<(), ErrorPayload> {
    let models = state.catalogue.lock().await.clone();
    protocol::write_plain(socket, &Response::Models { models }).await
}

pub(super) async fn handle_benchmark(
    socket: &mut TcpStream,
    size: u32,
) -> Result<(), ErrorPayload> {
    let size = size.clamp(4 * 1024, 512 * 1024) as usize;
    protocol::write_plain(socket, &Response::Benchmark { size: size as u32 }).await?;
    let incoming = protocol::read_bytes(socket).await?;
    if incoming.len() != size {
        return Err(ErrorPayload::new(
            "benchmark_size",
            "The peer benchmark payload did not match.",
            None,
        ));
    }
    protocol::write_bytes(socket, &incoming).await
}

async fn ensure_worker(state: &ServerState) -> Result<SocketAddr, ErrorPayload> {
    let mut worker = state.worker.lock().await;
    let running = match worker.as_mut() {
        Some(child) => child
            .child
            .try_wait()
            .map_err(protocol::io_error)?
            .is_none(),
        None => false,
    };
    if running {
        if let Some(target) = *state.rpc_target.lock().await {
            return Ok(target);
        }
    }
    if let Some(target) = state.rpc_override {
        *state.rpc_target.lock().await = Some(target);
        return Ok(target);
    }
    *worker = None;
    *state.rpc_target.lock().await = None;
    let Some(binary) = state.rpc_binary.clone() else {
        return Err(ErrorPayload::new(
            "rpc_worker_missing",
            "The pinned ggml-rpc-server is not installed.",
            Some("Install the runtime on this computer.".into()),
        ));
    };
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .map_err(protocol::io_error)?;
    let target = listener.local_addr().map_err(protocol::io_error)?;
    drop(listener);
    let job = ProcessJob::new();
    let mut child = Command::new(&binary)
        .args(["--host", "127.0.0.1", "--port", &target.port().to_string()])
        .spawn()
        .map_err(protocol::io_error)?;
    if let Some(job) = job.as_ref() {
        job.assign(&child)?;
    }
    if child.try_wait().map_err(protocol::io_error)?.is_some() {
        return Err(ErrorPayload::new(
            "rpc_worker_exited",
            "ggml-rpc-server exited before the tunnel opened.",
            Some("Inspect the runtime logs and retry.".into()),
        ));
    }
    *worker = Some(ManagedChild { child, _job: job });
    *state.rpc_target.lock().await = Some(target);
    Ok(target)
}

async fn connect_rpc(target: SocketAddr) -> Result<TcpStream, ErrorPayload> {
    if !target.ip().is_loopback() {
        return Err(ErrorPayload::new(
            "rpc_target_rejected",
            "The worker RPC target must be loopback.",
            None,
        ));
    }
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

pub(super) async fn handle_proxy_chat(
    socket: &mut TcpStream,
    state: &ServerState,
    request: Request,
) -> Result<(), ErrorPayload> {
    let Request::ProxyChat {
        messages,
        settings,
        images,
    } = request
    else {
        return Err(ErrorPayload::new(
            "peer_protocol",
            "The peer returned an unexpected response.",
            None,
        ));
    };
    let (api_key, api_port) = {
        let key = state.api_key.lock().await.clone();
        let port = *state.api_port.lock().await;
        (key, port)
    };
    let content =
        crate::commands::chat::complete_local(api_port, &api_key, messages, settings, images)
            .await?;
    protocol::write_plain(socket, &Response::ProxyChat { content }).await
}
