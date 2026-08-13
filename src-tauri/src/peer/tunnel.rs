use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
};

use super::{client::PeerClient, protocol};
use crate::types::ErrorPayload;

const TUNNEL_CHUNK: usize = 32 * 1024;

pub struct RpcForwarder {
    local_address: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl RpcForwarder {
    pub(crate) async fn start(client: Arc<PeerClient>) -> Result<Self, ErrorPayload> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .map_err(protocol::io_error)?;
        let local_address = listener.local_addr().map_err(protocol::io_error)?;
        let (stop, mut stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stopped => break,
                    accepted = listener.accept() => {
                        let Ok((local, _)) = accepted else { break };
                        let client = client.clone();
                        tokio::spawn(async move {
                            if let Ok((peer, noise)) = client.open_rpc_stream().await { let _ = client_bridge(local, peer, noise).await; }
                        });
                    }
                }
            }
        });
        Ok(Self {
            local_address,
            stop: Some(stop),
            task,
        })
    }
    pub fn local_address(&self) -> SocketAddr {
        self.local_address
    }
    pub async fn shutdown(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let _ = self.task.await;
    }
}

async fn client_bridge(
    mut local: TcpStream,
    mut peer: TcpStream,
    mut noise: snow::TransportState,
) -> Result<(), ErrorPayload> {
    let (mut local_read, mut local_write) = local.split();
    let (mut peer_read, mut peer_write) = peer.split();
    let mut from_local = [0_u8; TUNNEL_CHUNK];
    let mut encrypted = vec![0_u8; TUNNEL_CHUNK + 16];
    let mut from_peer = vec![0_u8; TUNNEL_CHUNK + 64];
    loop {
        tokio::select! {
            count = local_read.read(&mut from_local) => {
                let count = count.map_err(protocol::io_error)?;
                if count == 0 { break; }
                let encrypted_count = noise.write_message(&from_local[..count], &mut encrypted).map_err(super::crypto::noise_error)?;
                peer_write.write_u32(encrypted_count as u32).await.map_err(protocol::io_error)?;
                peer_write.write_all(&encrypted[..encrypted_count]).await.map_err(protocol::io_error)?;
            }
            frame_size = peer_read.read_u32() => {
                let frame_size = frame_size.map_err(protocol::io_error)? as usize;
                if frame_size > TUNNEL_CHUNK + 16 { return Err(ErrorPayload::new("tunnel_frame_large", "RPC tunnel frame exceeded its safety limit.", None)); }
                peer_read.read_exact(&mut from_peer[..frame_size]).await.map_err(protocol::io_error)?;
                let plain_count = noise.read_message(&from_peer[..frame_size], &mut from_local).map_err(super::crypto::noise_error)?;
                local_write.write_all(&from_local[..plain_count]).await.map_err(protocol::io_error)?;
            }
        }
    }
    Ok(())
}

pub(crate) async fn serve_rpc(
    mut peer: TcpStream,
    mut noise: snow::TransportState,
    mut rpc: TcpStream,
) -> Result<(), ErrorPayload> {
    let (mut peer_read, mut peer_write) = peer.split();
    let (mut rpc_read, mut rpc_write) = rpc.split();
    let mut encrypted = vec![0_u8; TUNNEL_CHUNK + 16];
    let mut plain = [0_u8; TUNNEL_CHUNK];
    loop {
        tokio::select! {
            frame_size = peer_read.read_u32() => {
                let frame_size = frame_size.map_err(protocol::io_error)? as usize;
                if frame_size > encrypted.len() { return Err(ErrorPayload::new("tunnel_frame_large", "RPC tunnel frame exceeded its safety limit.", None)); }
                peer_read.read_exact(&mut encrypted[..frame_size]).await.map_err(protocol::io_error)?;
                let count = noise.read_message(&encrypted[..frame_size], &mut plain).map_err(super::crypto::noise_error)?;
                rpc_write.write_all(&plain[..count]).await.map_err(protocol::io_error)?;
            }
            count = rpc_read.read(&mut plain) => {
                let count = count.map_err(protocol::io_error)?;
                if count == 0 { break; }
                let encrypted_count = noise.write_message(&plain[..count], &mut encrypted).map_err(super::crypto::noise_error)?;
                peer_write.write_u32(encrypted_count as u32).await.map_err(protocol::io_error)?;
                peer_write.write_all(&encrypted[..encrypted_count]).await.map_err(protocol::io_error)?;
            }
        }
    }
    Ok(())
}
