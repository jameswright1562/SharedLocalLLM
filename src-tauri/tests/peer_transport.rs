use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use serde_json::json;
use shared_local_llm::peer::{PeerClient, PeerServer, PeerServerConfig, TrustedPeer};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

async fn start_peer(code: &str, rpc_override: Option<SocketAddr>) -> PeerServer {
    PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        device_id: "worker-id".into(),
        device_name: "Worker".into(),
        capabilities: json!({"gpu":"test"}),
        pairing_code: Some(code.into()),
        trusted_peers: vec![],
        rpc_binary: None,
        rpc_override,
        catalogue: json!([]),
        api_key: "test-key".into(),
        api_port: 11435,
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn rejects_wrong_pairing_code_and_accepts_correct_code() {
    let peer = start_peer("482916", None).await;
    let mut pairing_completion = peer.pairing_completion();

    assert!(PeerClient::pair(
        peer.address(),
        "000000",
        "client-id",
        "Client",
        json!({"id":"client-id","name":"Client"}),
    )
    .await
    .is_err());
    let client = PeerClient::pair(
        peer.address(),
        "482 916",
        "client-id",
        "Client",
        json!({"id":"client-id","name":"Client","gpu":{"name":"Client GPU"}}),
    )
    .await
    .unwrap();
    assert_eq!(client.capabilities().await.unwrap()["gpu"], "test");
    tokio::time::timeout(Duration::from_secs(1), pairing_completion.changed())
        .await
        .unwrap()
        .unwrap();
    let paired = pairing_completion.borrow().clone().unwrap();
    assert_eq!(paired.device_id, "client-id");
    assert_eq!(paired.device_name, "Client");
    assert_eq!(paired.capabilities["gpu"]["name"], "Client GPU");
    assert_eq!(paired.channel_key, client.channel_key());
    assert!(client.heartbeat().await.unwrap());
    peer.shutdown().await;
}

#[tokio::test]
async fn benchmarks_encrypted_bidirectional_messages() {
    let peer = start_peer("123456", None).await;
    let client = PeerClient::pair(
        peer.address(),
        "123456",
        "client-id",
        "Client",
        json!({"id":"client-id","name":"Client"}),
    )
    .await
    .unwrap();

    let result = client.benchmark(32 * 1024, 4).await.unwrap();

    assert!(result.throughput_mbps > 0.0);
    assert_eq!(result.samples, 4);
    assert!(result.latency_p95_ms >= result.latency_median_ms);
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
    let peer = start_peer("654321", Some(echo_address)).await;
    let client = Arc::new(
        PeerClient::pair(
            peer.address(),
            "654321",
            "client-id",
            "Client",
            json!({"id":"client-id","name":"Client"}),
        )
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
async fn trusted_peer_reconnects_after_the_worker_listener_restarts() {
    let first = start_peer("741852", None).await;
    let paired = PeerClient::pair(
        first.address(),
        "741852",
        "client-id",
        "Client",
        json!({"id":"client-id","name":"Client"}),
    )
    .await
    .unwrap();
    let channel_key = paired.channel_key().to_owned();
    first.shutdown().await;

    let restarted = PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        device_id: "worker-id".into(),
        device_name: "Worker".into(),
        capabilities: json!({"id":"worker-id","name":"Worker"}),
        pairing_code: None,
        trusted_peers: vec![TrustedPeer {
            device_id: "client-id".into(),
            device_name: "Client".into(),
            channel_key: channel_key.clone(),
        }],
        rpc_binary: None,
        rpc_override: None,
        catalogue: json!([]),
        api_key: "test-key".into(),
        api_port: 11435,
    })
    .await
    .unwrap();
    let reconnected = PeerClient::trusted(restarted.address(), channel_key, "client-id".into());

    assert!(reconnected.heartbeat().await.unwrap());
    restarted.shutdown().await;
}
