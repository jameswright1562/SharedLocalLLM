use serde::{Deserialize, Serialize};

use crate::types::{ErrorPayload, NetworkBenchmark};

#[cfg(windows)]
pub fn windows_network_profile() -> Option<String> {
    let script = "@(Get-NetConnectionProfile | Where-Object IPv4Connectivity -ne 'Disconnected' | Select-Object -ExpandProperty NetworkCategory | Sort-Object -Unique) -join ', '";
    crate::hardware::output_with_timeout(
        "powershell.exe",
        &["-NoProfile", "-NonInteractive", "-Command", script],
        std::time::Duration::from_secs(4),
    )
    .map(|value| value.trim().to_string())
    .filter(|value| !value.is_empty())
}

#[cfg(not(windows))]
pub fn windows_network_profile() -> Option<String> {
    None
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub throughput_mbps: f64,
    pub latency_p95_ms: f64,
    pub jitter_ms: f64,
    pub packet_loss_percent: f64,
}

impl NetworkMetrics {
    pub fn new(
        throughput_mbps: f64,
        latency_p95_ms: f64,
        jitter_ms: f64,
        packet_loss_percent: f64,
    ) -> Self {
        Self {
            throughput_mbps,
            latency_p95_ms,
            jitter_ms,
            packet_loss_percent,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NetworkClass {
    Good,
    Usable,
    Poor,
}

pub fn classify_network(metrics: NetworkMetrics) -> NetworkClass {
    if metrics.latency_p95_ms <= 3.0 {
        NetworkClass::Good
    } else if metrics.latency_p95_ms <= 10.0 {
        NetworkClass::Usable
    } else {
        NetworkClass::Poor
    }
}

pub async fn benchmark_peer(
    peer_address: Option<&str>,
    adapter: &str,
) -> Result<NetworkBenchmark, ErrorPayload> {
    let address = peer_address.ok_or_else(|| {
        ErrorPayload::new(
            "peer_unavailable",
            "Pair a reachable computer before running the encrypted network benchmark.",
            Some("Open Nodes, generate a pairing code, and connect the second computer.".into()),
        )
    })?;
    let started = std::time::Instant::now();
    let stream = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        tokio::net::TcpStream::connect(address),
    )
    .await
    .map_err(|_| {
        ErrorPayload::new(
            "network_timeout",
            "The peer did not answer within three seconds.",
            Some("Check the link and firewall settings.".into()),
        )
    })?
    .map_err(|e| {
        ErrorPayload::new(
            "peer_unreachable",
            e.to_string(),
            Some("Verify the peer address and that SharedLocalLLM is running.".into()),
        )
    })?;
    drop(stream);
    let latency = started.elapsed().as_secs_f64() * 1000.0;
    let metrics = NetworkMetrics::new(0.0, latency, 0.0, 0.0);
    Ok(NetworkBenchmark {
        down_mbps: 0.0,
        up_mbps: 0.0,
        latency_median_ms: latency,
        latency_p95_ms: latency,
        jitter_ms: 0.0,
        packet_loss_percent: 0.0,
        classification: format!("{:?}", classify_network(metrics)).to_lowercase(),
        adapter: format!("{adapter} · connectivity-only probe; throughput agent unavailable"),
        windows_profile: windows_network_profile(),
    })
}
