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

async fn start_peer(rpc_override: Option<SocketAddr>) -> PeerServer {
    PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        device_id: "worker-id".into(),
        device_name: "Worker".into(),
        capabilities: json!({"id":"worker-id","name":"Worker","gpu":{"name":"Worker GPU"}}),
        rpc_binary: None,
        rpc_override,
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
    let echo_task = tokio::spawn(async move {
        let (mut socket, _) = echo.accept().await.unwrap();
        let mut buffer = [0_u8; 256];
        loop {
            let count = socket.read(&mut buffer).await.unwrap();
            if count == 0 {
                break;
            }
            socket.write_all(&buffer[..count]).await.unwrap();
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
