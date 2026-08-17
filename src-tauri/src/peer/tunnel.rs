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
        let _ = listener.set_ttl(64);
        let local_address = listener.local_addr().map_err(protocol::io_error)?;
        let (stop, mut stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stopped => break,
                    accepted = listener.accept() => {
                        let Ok((local, _)) = accepted else { break };
                        let _ = local.set_nodelay(true);
                        let client = client.clone();
                        tokio::spawn(async move {
                            match client.open_rpc_stream().await {
                                Ok(peer) => { let _ = bridge(peer, local).await; }
                                Err(error) => eprintln!(
                                    "{} WARN rpc_tunnel_open_failed: {}",
                                    crate::pairing::now(),
                                    error
                                ),
                            }
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

/// Bidirectionally stream length-prefixed byte frames between two sockets.
pub(crate) async fn bridge(mut left: TcpStream, mut right: TcpStream) -> Result<(), ErrorPayload> {
    let _ = left.set_nodelay(true);
    let _ = right.set_nodelay(true);
    let (mut left_read, mut left_write) = left.split();
    let (mut right_read, mut right_write) = right.split();
    let mut buffer = vec![0_u8; TUNNEL_CHUNK];
    loop {
        tokio::select! {
            frame_size = left_read.read_u32() => {
                let frame_size = frame_size.map_err(protocol::io_error)? as usize;
                if frame_size > TUNNEL_CHUNK {
                    return Err(ErrorPayload::new(
                        "tunnel_frame_large",
                        "RPC tunnel frame exceeded its safety limit.",
                        None,
                    ));
                }
                left_read.read_exact(&mut buffer[..frame_size]).await.map_err(protocol::io_error)?;
                right_write.write_all(&buffer[..frame_size]).await.map_err(protocol::io_error)?;
                right_write.flush().await.map_err(protocol::io_error)?;
            }
            count = right_read.read(&mut buffer) => {
                let count = count.map_err(protocol::io_error)?;
                if count == 0 { break; }
                left_write.write_u32(count as u32).await.map_err(protocol::io_error)?;
                left_write.write_all(&buffer[..count]).await.map_err(protocol::io_error)?;
                left_write.flush().await.map_err(protocol::io_error)?;
            }
        }
    }
    Ok(())
}
