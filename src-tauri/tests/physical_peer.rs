//! Physical two-PC acceptance tests for the SharedLocalLLM peer channel.
//!
//! These tests are `#[ignore]`d because they require a real second Windows computer running the
//! SharedLocalLLM app and a working link between the two machines. They are still compiled and
//! type-checked by `cargo test --all-targets` and `cargo clippy`; they only execute when explicitly
//! requested with `--ignored`.
//!
//! # Prerequisites
//!
//! Start the SharedLocalLLM app on the peer PC. Its firewall rules must allow inbound TCP 49158
//! (`SharedLocalLLM`) and UDP 49157 (`SharedLocalLLM Discovery`), both `-Profile Any`.
//!
//! # Running
//!
//! On this PC, set the peer endpoint and (for the heartbeat test) the shared channel key shown by
//! the app, then run:
//!
//! ```text
//! $env:SHARED_LOCAL_LLM_PEER_ENDPOINT = "10.10.10.2"
//! $env:SHARED_LOCAL_LLM_PEER_CHANNEL_KEY = "<channel key shown by the app on the peer PC>"
//! cargo test --manifest-path src-tauri/Cargo.toml --test physical_peer -- --ignored --nocapture
//! ```
//!
//! `SHARED_LOCAL_LLM_PEER_ENDPOINT` accepts either a bare IP (`10.10.10.2`) or an explicit
//! `IP:port` (`10.10.10.2:49158`); when the port is omitted the default peer port 49158 is used.
//!
//! These are network/firewall/loopback smoke checks, not proof of distributed GPU inference; that
//! remains the manual two-computer acceptance matrix in `docs/testing.md`.

use std::{
    net::{IpAddr, SocketAddr},
    time::Duration,
};

use shared_local_llm::peer::PeerClient;
use tokio::net::TcpStream;

const PEER_PORT: u16 = 49_158;

fn peer_endpoint() -> SocketAddr {
    let raw = std::env::var("SHARED_LOCAL_LLM_PEER_ENDPOINT").unwrap_or_else(|_| {
        panic!(
            "set SHARED_LOCAL_LLM_PEER_ENDPOINT to the peer IP (for example 10.10.10.2) or \
             IP:port (for example 10.10.10.2:49158) before running the physical peer tests"
        )
    });
    raw.parse::<SocketAddr>()
        .or_else(|_| {
            raw.parse::<IpAddr>()
                .map(|ip| SocketAddr::from((ip, PEER_PORT)))
        })
        .unwrap_or_else(|_| {
            panic!(
                "SHARED_LOCAL_LLM_PEER_ENDPOINT '{raw}' is not a valid IP or IP:port; expected \
                 something like '10.10.10.2' or '10.10.10.2:49158'"
            )
        })
}

#[tokio::test]
#[ignore]
async fn peer_is_tcp_reachable_over_the_cable() {
    let endpoint = peer_endpoint();
    tokio::time::timeout(Duration::from_secs(3), TcpStream::connect(endpoint))
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {endpoint} to accept a TCP connection"))
        .unwrap_or_else(|error| panic!("failed to connect to {endpoint}: {error}"));
}

#[tokio::test]
#[ignore]
async fn peer_broadcasts_discovery_announcements() {
    let announcements = shared_local_llm::peer::discover(Duration::from_secs(10))
        .await
        .unwrap_or_else(|error| panic!("discovery on UDP port 49157 failed: {error}"));
    assert!(
        !announcements.is_empty(),
        "no discovery announcements received on UDP port 49157 within 10 seconds"
    );
    for (announcement, source) in &announcements {
        assert!(
            !announcement.device_id.is_empty(),
            "announcement from {source} has an empty device_id"
        );
        assert_eq!(
            announcement.peer_port, PEER_PORT,
            "announcement from {source} advertises an unexpected peer port"
        );
    }
}

#[tokio::test]
#[ignore]
async fn peer_channel_completes_a_heartbeat_with_a_shared_channel_key() {
    let endpoint = peer_endpoint();
    let channel_key = std::env::var("SHARED_LOCAL_LLM_PEER_CHANNEL_KEY").unwrap_or_else(|_| {
        panic!(
            "set SHARED_LOCAL_LLM_PEER_CHANNEL_KEY to the channel key shown by the app on the \
             peer PC before running this test"
        )
    });
    let client = PeerClient::trusted(endpoint, channel_key, "physical-test".into());
    let healthy = client
        .heartbeat()
        .await
        .unwrap_or_else(|error| panic!("heartbeat over the peer channel failed: {error}"));
    assert!(healthy, "peer answered but did not confirm the heartbeat");
}
