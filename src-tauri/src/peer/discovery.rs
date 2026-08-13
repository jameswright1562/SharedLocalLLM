use std::{
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::{net::UdpSocket, sync::oneshot, task::JoinHandle};

use crate::types::ErrorPayload;

const DISCOVERY_PORT: u16 = 49_157;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryAnnouncement {
    pub protocol_version: u16,
    pub device_id: String,
    pub device_name: String,
    pub peer_port: u16,
}

pub struct DiscoveryBroadcaster {
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl DiscoveryBroadcaster {
    pub async fn start(announcement: DiscoveryAnnouncement) -> Result<Self, ErrorPayload> {
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
            .await
            .map_err(io_error)?;
        socket.set_broadcast(true).map_err(io_error)?;
        let message = serde_json::to_vec(&announcement)
            .map_err(|error| ErrorPayload::new("discovery_encode", error.to_string(), None))?;
        let (stop, mut stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            let target = SocketAddr::from((Ipv4Addr::BROADCAST, DISCOVERY_PORT));
            loop {
                tokio::select! {
                    _ = &mut stopped => break,
                    _ = tokio::time::sleep(Duration::from_secs(2)) => { let _ = socket.send_to(&message, target).await; }
                }
            }
        });
        Ok(Self {
            stop: Some(stop),
            task,
        })
    }
    pub async fn shutdown(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let _ = self.task.await;
    }
}

pub async fn discover(
    timeout: Duration,
) -> Result<Vec<(DiscoveryAnnouncement, SocketAddr)>, ErrorPayload> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, DISCOVERY_PORT))
        .await
        .map_err(io_error)?;
    let deadline = tokio::time::Instant::now() + timeout;
    let mut peers = Vec::new();
    let mut buffer = [0u8; 2048];
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        let Ok(Ok((count, source))) =
            tokio::time::timeout(remaining, socket.recv_from(&mut buffer)).await
        else {
            break;
        };
        if let Ok(value) = serde_json::from_slice::<DiscoveryAnnouncement>(&buffer[..count]) {
            let peer_port = value.peer_port;
            peers.push((value, SocketAddr::new(source.ip(), peer_port)));
        }
    }
    Ok(peers)
}

fn io_error(error: std::io::Error) -> ErrorPayload {
    ErrorPayload::new(
        "discovery_io",
        error.to_string(),
        Some("Allow SharedLocalLLM on Private networks or enter the peer address manually.".into()),
    )
}

#[cfg(test)]
mod tests {
    use super::directed_broadcast;
    use std::net::Ipv4Addr;

    #[test]
    fn calculates_a_directed_broadcast_for_private_and_link_local_interfaces() {
        assert_eq!(
            directed_broadcast(
                Ipv4Addr::new(192, 168, 50, 12),
                Ipv4Addr::new(255, 255, 255, 0)
            ),
            Ipv4Addr::new(192, 168, 50, 255)
        );
        assert_eq!(
            directed_broadcast(
                Ipv4Addr::new(169, 254, 179, 236),
                Ipv4Addr::new(255, 255, 0, 0)
            ),
            Ipv4Addr::new(169, 254, 255, 255)
        );
    }
}
