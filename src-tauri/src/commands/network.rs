use tauri::State;

use crate::{
    network::{classify_network, NetworkClass, NetworkMetrics},
    state::AppState,
    types::{ErrorPayload, NetworkBenchmark},
};

#[tauri::command]
pub async fn run_network_test(
    state: State<'_, AppState>,
) -> Result<NetworkBenchmark, ErrorPayload> {
    let client = state.peer_client().await?;
    if !client.heartbeat().await? {
        return Err(ErrorPayload::new(
            "peer_unavailable",
            "The paired computer did not answer the heartbeat.",
            None,
        ));
    }
    let result = client.benchmark(256 * 1024, 4).await?;
    let metrics = NetworkMetrics::new(result.throughput_mbps, result.latency_p95_ms, 0.0, 0.0);
    let classification = match classify_network(metrics) {
        NetworkClass::Good => "good",
        NetworkClass::Usable => "usable",
        NetworkClass::Poor => "poor",
    };
    let adapter = state.lock()?.local.adapter.name.clone();
    let benchmark = NetworkBenchmark {
        down_mbps: result.throughput_mbps,
        up_mbps: result.throughput_mbps,
        latency_median_ms: result.latency_median_ms,
        latency_p95_ms: result.latency_p95_ms,
        jitter_ms: -1.0,
        packet_loss_percent: -1.0,
        classification: classification.into(),
        adapter: format!("{adapter} · encrypted peer channel (round-trip)"),
    };
    state.lock()?.network = Some(benchmark.clone());
    state.log(
        "INFO",
        "network_benchmark_completed",
        &format!(
            "classification={} throughput_mbps={:.1} p95_ms={:.2}",
            benchmark.classification, benchmark.down_mbps, benchmark.latency_p95_ms
        ),
    );
    Ok(benchmark)
}
