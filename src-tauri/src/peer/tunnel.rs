use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{
        tcp::{OwnedReadHalf, OwnedWriteHalf},
        TcpListener, TcpStream,
    },
    sync::oneshot,
    task::JoinHandle,
};

use super::{client::PeerClient, protocol};
use crate::types::ErrorPayload;

const TUNNEL_CHUNK: usize = 32 * 1024;
const RPC_FORWARD_PORT: u16 = 50053;

pub struct RpcForwarder {
    local_address: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl RpcForwarder {
    pub(crate) async fn start(
        client: Arc<PeerClient>,
        use_new_library: bool,
    ) -> Result<Self, ErrorPayload> {
        let listener = TcpListener::bind((
            Ipv4Addr::LOCALHOST,
            if use_new_library { RPC_FORWARD_PORT } else { 0 },
        ))
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
///
/// Each direction is pumped by its own task that owns one half of each socket
/// and its own buffer. Because a task never awaits inside a `tokio::select!`,
/// an in-flight framed read (`read_u32`/`read_exact`) is never cancelled, so
/// the length-prefixed framing can never desynchronize. When either direction
/// finishes (peer closed or error), the other is aborted to tear the tunnel
/// down cleanly.
pub(crate) async fn bridge(left: TcpStream, right: TcpStream) -> Result<(), ErrorPayload> {
    let _ = left.set_nodelay(true);
    let _ = right.set_nodelay(true);
    let (left_read, left_write) = left.into_split();
    let (right_read, right_write) = right.into_split();

    // left -> right: unframe length-prefixed frames from the peer tunnel and
    // write the raw bytes to the local RPC socket.
    let mut deframe = tokio::spawn(pump_frames_to_raw(left_read, right_write));
    // right -> left: read raw bytes from the local RPC socket and forward them
    // as length-prefixed frames over the peer tunnel.
    let mut enframe = tokio::spawn(pump_raw_to_frames(right_read, left_write));

    let outcome = tokio::select! {
        result = &mut deframe => {
            enframe.abort();
            result
        }
        result = &mut enframe => {
            deframe.abort();
            result
        }
    };

    match outcome {
        Ok(result) => result,
        Err(join_error) if join_error.is_cancelled() => Ok(()),
        Err(join_error) => Err(ErrorPayload::new(
            "tunnel_task_failed",
            join_error.to_string(),
            None,
        )),
    }
}

/// Read length-prefixed frames from `source` and write the raw payloads to
/// `sink`. Runs to completion without any cancellation point mid-frame.
async fn pump_frames_to_raw(
    mut source: OwnedReadHalf,
    mut sink: OwnedWriteHalf,
) -> Result<(), ErrorPayload> {
    let mut buffer = vec![0_u8; TUNNEL_CHUNK];
    loop {
        let frame_size = match source.read_u32().await {
            Ok(size) => size as usize,
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(protocol::io_error(error)),
        };
        if frame_size > TUNNEL_CHUNK {
            return Err(ErrorPayload::new(
                "tunnel_frame_large",
                "RPC tunnel frame exceeded its safety limit.",
                None,
            ));
        }
        source
            .read_exact(&mut buffer[..frame_size])
            .await
            .map_err(protocol::io_error)?;
        sink.write_all(&buffer[..frame_size])
            .await
            .map_err(protocol::io_error)?;
        sink.flush().await.map_err(protocol::io_error)?;
    }
    Ok(())
}

/// Read raw bytes from `source` and forward them as length-prefixed frames to
/// `sink`. Runs to completion without any cancellation point mid-frame.
async fn pump_raw_to_frames(
    mut source: OwnedReadHalf,
    mut sink: OwnedWriteHalf,
) -> Result<(), ErrorPayload> {
    let mut buffer = vec![0_u8; TUNNEL_CHUNK];
    loop {
        let count = source.read(&mut buffer).await.map_err(protocol::io_error)?;
        if count == 0 {
            break;
        }
        sink.write_u32(count as u32)
            .await
            .map_err(protocol::io_error)?;
        sink.write_all(&buffer[..count])
            .await
            .map_err(protocol::io_error)?;
        sink.flush().await.map_err(protocol::io_error)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn connected_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind loopback listener");
        let address = listener.local_addr().expect("listener address");
        let client = TcpStream::connect(address)
            .await
            .expect("connect loopback client");
        let (server, _) = listener.accept().await.expect("accept loopback client");
        (client, server)
    }

    /// Simple deterministic xorshift so the "random" frame sizes are
    /// reproducible without pulling in an RNG dependency.
    fn next_random(state: &mut u64) -> u64 {
        let mut value = *state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        *state = value;
        value
    }

    /// Drive simultaneous bidirectional traffic through `bridge` and assert
    /// every byte survives, in order, in both directions.
    ///
    /// `left_peer` speaks the length-prefixed framing (what a remote peer
    /// tunnel does); `right_peer` speaks raw bytes (what the local RPC socket
    /// does). We know the exact byte totals in advance, so each side reads
    /// exactly that many bytes before closing — this keeps the abort-on-close
    /// behaviour from truncating data mid-test.
    async fn run_roundtrip(frame_count: usize, mut seed: u64) {
        let (left, left_peer) = connected_pair().await;
        let (right, right_peer) = connected_pair().await;

        // Build the payloads both directions will send.
        let mut left_to_right_frames: Vec<Vec<u8>> = Vec::with_capacity(frame_count);
        let mut right_to_left_chunks: Vec<Vec<u8>> = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let l_len = (next_random(&mut seed) as usize % 64) + 1;
            let r_len = (next_random(&mut seed) as usize % 64) + 1;
            left_to_right_frames.push(vec![(i % 251) as u8; l_len]);
            right_to_left_chunks.push(vec![((i + 128) % 251) as u8; r_len]);
        }
        let expected_left_to_right: Vec<u8> =
            left_to_right_frames.iter().flatten().copied().collect();
        let expected_right_to_left: Vec<u8> =
            right_to_left_chunks.iter().flatten().copied().collect();
        let left_to_right_total = expected_left_to_right.len();
        let right_to_left_total = expected_right_to_left.len();

        let bridge_task = tokio::spawn(async move { bridge(left, right).await });

        let (mut left_read, mut left_write) = left_peer.into_split();
        let (mut right_read, mut right_write) = right_peer.into_split();

        // left_peer writes framed messages destined for right_peer.
        let left_writer = tokio::spawn(async move {
            for frame in &left_to_right_frames {
                left_write.write_u32(frame.len() as u32).await.unwrap();
                left_write.write_all(frame).await.unwrap();
                left_write.flush().await.unwrap();
            }
            left_write
        });

        // right_peer writes raw bytes destined for left_peer.
        let right_writer = tokio::spawn(async move {
            for chunk in &right_to_left_chunks {
                right_write.write_all(chunk).await.unwrap();
                right_write.flush().await.unwrap();
            }
            right_write
        });

        // right_peer reads the raw stream and expects the concatenated frames.
        let right_reader = tokio::spawn(async move {
            let mut received = vec![0_u8; left_to_right_total];
            right_read.read_exact(&mut received).await.unwrap();
            received
        });

        // left_peer reads framed messages and reassembles the raw byte stream.
        let left_reader = tokio::spawn(async move {
            let mut received = Vec::with_capacity(right_to_left_total);
            while received.len() < right_to_left_total {
                let size = left_read.read_u32().await.unwrap() as usize;
                let mut frame = vec![0_u8; size];
                left_read.read_exact(&mut frame).await.unwrap();
                received.extend_from_slice(&frame);
            }
            received
        });

        let right_received = right_reader.await.unwrap();
        let left_received = left_reader.await.unwrap();
        assert_eq!(right_received, expected_left_to_right);
        assert_eq!(left_received, expected_right_to_left);

        // Closing both ends lets the bridge tear down cleanly.
        drop(left_writer.await.unwrap());
        drop(right_writer.await.unwrap());
        let _ = bridge_task.await.unwrap();
    }

    #[tokio::test]
    async fn bridge_preserves_bidirectional_traffic() {
        run_roundtrip(200, 0x1234_5678_9abc_def0).await;
    }

    #[tokio::test]
    async fn bridge_survives_random_small_frame_stress() {
        for iteration in 0..25_u64 {
            run_roundtrip(64, 0xdead_beef_0000_0001 ^ (iteration.wrapping_mul(0x9e37_79b9))).await;
        }
    }
}
