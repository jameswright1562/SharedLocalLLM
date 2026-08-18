use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use serde_json::json;
use shared_local_llm::peer::{PeerClient, PeerServer, PeerServerConfig};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

async fn start_peer(rpc_target: Option<SocketAddr>) -> PeerServer {
    PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        device_id: "worker-id".into(),
        device_name: "Worker".into(),
        capabilities: json!({"id":"worker-id","name":"Worker","gpu":{"name":"Worker GPU"}}),
        rpc_target: rpc_target.unwrap_or_else(|| SocketAddr::from((Ipv4Addr::LOCALHOST, 50052))),
        catalogue: json!([]),
        api_key: "test-key".into(),
        api_port: 11435,
    })
    .await
    .unwrap()
}

fn client(endpoint: SocketAddr) -> PeerClient {
    PeerClient::new(endpoint, "client-id".into())
}

#[tokio::test]
async fn connects_and_serves_heartbeat_and_capabilities() {
    let peer = start_peer(None).await;
    let client = client(peer.address())
        .connect(
            "Client",
            json!({"id":"client-id","name":"Client","gpu":{"name":"Client GPU"}}),
        )
        .await
        .unwrap();
    assert_eq!(client.remote_device_id(), "worker-id");
    assert_eq!(client.remote_device_name(), "Worker");
    assert_eq!(
        client.capabilities().await.unwrap()["gpu"]["name"],
        "Worker GPU"
    );
    assert!(client.heartbeat().await.unwrap());
    peer.shutdown().await;
}

#[tokio::test]
async fn benchmarks_bidirectional_messages() {
    let peer = start_peer(None).await;
    let client = client(peer.address())
        .connect("Client", json!({"id":"client-id","name":"Client"}))
        .await
        .unwrap();

    let result = client.benchmark(32 * 1024, 4).await.unwrap();

    assert!(result.throughput_mbps > 0.0);
    assert_eq!(result.samples, 4);
    assert!(result.latency_p95_ms >= result.latency_median_ms);
    peer.shutdown().await;
}

#[tokio::test]
async fn benchmark_handles_the_full_network_test_payload_size() {
    let peer = start_peer(None).await;
    let client = client(peer.address())
        .connect("Client", json!({"id":"client-id","name":"Client"}))
        .await
        .unwrap();

    let result = client.benchmark(256 * 1024, 4).await.unwrap();

    assert!(result.throughput_mbps > 0.0);
    assert_eq!(result.samples, 4);
    peer.shutdown().await;
}

#[tokio::test]
async fn rpc_tunnel_preserves_bytes_and_cleans_up_after_disconnect() {
    let echo = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let echo_address = echo.local_addr().unwrap();
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
    let echo_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                accepted = echo.accept() => {
                    let Ok((mut socket, _)) = accepted else { break };
                    tokio::spawn(async move {
                        let mut buffer = [0_u8; 256];
                        loop {
                            match socket.read(&mut buffer).await {
                                Ok(0) | Err(_) => break,
                                Ok(count) => {
                                    if socket.write_all(&buffer[..count]).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    });
                }
            }
        }
    });
    let peer = start_peer(Some(echo_address)).await;
    let client = Arc::new(
        client(peer.address())
            .connect("Client", json!({"id":"client-id","name":"Client"}))
            .await
            .unwrap(),
    );
    let tunnel = client.start_rpc_forwarder().await.unwrap();
    let mut local = tokio::net::TcpStream::connect(tunnel.local_address())
        .await
        .unwrap();

    local.write_all(b"rpc tunnel integrity").await.unwrap();
    let mut result = vec![0; 20];
    local.read_exact(&mut result).await.unwrap();
    assert_eq!(result, b"rpc tunnel integrity");

    drop(local);
    tunnel.shutdown().await;
    let _ = stop_tx.send(());
    tokio::time::timeout(Duration::from_secs(2), echo_task)
        .await
        .unwrap()
        .unwrap();
    peer.shutdown().await;
}

#[tokio::test]
async fn client_reconnects_after_the_worker_listener_restarts() {
    let first = start_peer(None).await;
    let client = client(first.address())
        .connect("Client", json!({"id":"client-id","name":"Client"}))
        .await
        .unwrap();
    assert!(client.heartbeat().await.unwrap());
    first.shutdown().await;

    let restarted = start_peer(None).await;
    let reconnected = PeerClient::new(restarted.address(), "client-id".into());
    assert!(reconnected.heartbeat().await.unwrap());
    restarted.shutdown().await;
}
