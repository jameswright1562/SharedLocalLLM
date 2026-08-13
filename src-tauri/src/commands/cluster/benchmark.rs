use std::time::Instant;

use tauri::State;

use crate::{
    commands::pairing::require_private_network,
    runtime,
    state::AppState,
    types::{ErrorPayload, InferenceBenchmark, ModelLoadConfig},
};

use super::split;

#[tauri::command]
pub async fn run_inference_benchmark(
    model_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<InferenceBenchmark>, ErrorPayload> {
    let (model, nodes) = {
        let inner = state.lock()?;
        let model = inner
            .models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
            .ok_or_else(|| {
                ErrorPayload::new(
                    "model_not_found",
                    "The benchmark model is unavailable.",
                    None,
                )
            })?;
        (
            model,
            split::cluster_nodes(&inner.local, inner.peers.first()),
        )
    };
    let executable = runtime::runtime_root()
        .join("current")
        .join("llama-bench.exe");
    if !executable.is_file() {
        return Err(ErrorPayload::new(
            "benchmark_runtime_missing",
            "llama-bench.exe is not installed.",
            Some("Install or repair the pinned runtime.".into()),
        ));
    }
    let paired = nodes.len() == 2;
    let online_gpu_nodes = nodes
        .into_iter()
        .filter(|node| node.online && node.gpu.vram_available_gb > 0.0)
        .collect::<Vec<_>>();
    if paired && online_gpu_nodes.len() != 2 {
        return Err(ErrorPayload::new(
            "benchmark_peer_offline",
            "The paired computer must be online for a two-computer GPU benchmark.",
            Some(
                "Open SharedLocalLLM on both computers and wait for both nodes to show reachable."
                    .into(),
            ),
        ));
    }
    let total_layers = model.layer_count.ok_or_else(|| {
        ErrorPayload::new(
            "layer_metadata_missing",
            "This GGUF file does not report a layer count for a measured GPU split.",
            Some("Refresh the model catalogue or choose another model.".into()),
        )
    })?;
    let allocations = fit_gpu_layers(&model, total_layers, &online_gpu_nodes)?;
    let distributed = allocations.len() == 2;
    let forwarder = if distributed {
        require_private_network()?;
        let client = state.peer_client().await?;
        client.heartbeat().await?;
        Some(client.start_rpc_forwarder().await?)
    } else {
        None
    };
    let rpc_endpoint = forwarder
        .as_ref()
        .map(|forwarder| forwarder.local_address().to_string());
    let layer_counts = allocations
        .iter()
        .map(|allocation| allocation.layers)
        .collect::<Vec<_>>();
    let mut command = tokio::process::Command::new(executable);
    command
        .args(benchmark_arguments(
            &model.shard_paths[0],
            &layer_counts,
            rpc_endpoint.as_deref(),
        ))
        .kill_on_drop(true);
    state.log(
        "INFO",
        "benchmark_started",
        &format!(
            "topology={} gpu_layers={}",
            if distributed { "distributed" } else { "local" },
            layer_counts
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("/")
        ),
    );
    let (cancel, cancelled) = tokio::sync::oneshot::channel();
    *state
        .benchmark_cancel
        .lock()
        .map_err(|_| benchmark_state_error())? = Some(cancel);
    let started = Instant::now();
    let output = tokio::select! {
        output = command.output() => output.map_err(|error| ErrorPayload::new("benchmark_process", error.to_string(), None)),
        _ = cancelled => Err(ErrorPayload::new("benchmark_cancelled", "The inference benchmark was cancelled.", None)),
    };
    let _ = state.benchmark_cancel.lock().map(|mut slot| slot.take());
    if let Some(forwarder) = forwarder {
        forwarder.shutdown().await;
    }
    let output = output?;
    if !output.status.success() {
        let error = ErrorPayload::new(
            "benchmark_failed",
            crate::state::redact_diagnostic(String::from_utf8_lossy(&output.stderr).trim()),
            Some("Review the model fit recommendation and runtime logs.".into()),
        );
        state.log("ERROR", "benchmark_failed", &error.to_string());
        return Err(error);
    }
    let (prompt, generation) = parse_benchmark_rates(&output.stdout)?;
    let result = InferenceBenchmark {
        id: uuid::Uuid::new_v4().to_string(),
        model_name: model.name,
        topology: if distributed { "distributed" } else { "local" }.into(),
        gpu_layers: allocations,
        prompt_tokens_per_second: prompt,
        generation_tokens_per_second: generation,
        load_time_seconds: started.elapsed().as_secs_f64(),
        memory_peak_gb: 0.0,
        recommended: true,
        ran_at: format!("{}", crate::pairing::now()),
        error: None,
    };
    state.lock()?.benchmarks.push(result.clone());
    state.persist()?;
    state.log(
        "INFO",
        "benchmark_completed",
        &format!(
            "topology={} prompt_tps={prompt:.2} generation_tps={generation:.2}",
            result.topology
        ),
    );
    Ok(vec![result])
}

fn fit_gpu_layers(
    model: &crate::types::ModelRecord,
    total_layers: u32,
    nodes: &[crate::types::NodeCapabilities],
) -> Result<Vec<crate::types::GpuLayerAllocation>, ErrorPayload> {
    let minimum = nodes.len() as u32;
    for gpu_layers in (minimum..=total_layers).rev() {
        let allocations = split::distribute_layers_by_vram(gpu_layers, nodes);
        let load_config = ModelLoadConfig {
            context_size: 4096,
            gpu_layers: allocations.clone(),
            force: false,
        };
        if split::build_split_estimate(model, &load_config, nodes)
            .is_ok_and(|(estimate, _)| estimate.devices.iter().all(|device| device.fits))
        {
            return Ok(allocations);
        }
    }
    Err(ErrorPayload::new(
        "benchmark_split_exceeds_vram",
        "The available GPUs cannot each hold even one model layer for this benchmark.",
        Some("Close GPU-heavy applications or choose a smaller quantization.".into()),
    ))
}

fn benchmark_arguments(model_path: &str, layer_counts: &[u32], rpc: Option<&str>) -> Vec<String> {
    let total_layers = layer_counts.iter().sum::<u32>();
    let mut arguments = vec![
        "--model".into(),
        model_path.into(),
        "--n-prompt".into(),
        "512".into(),
        "--n-gen".into(),
        "128".into(),
        "--repetitions".into(),
        "3".into(),
        "--output".into(),
        "json".into(),
        "--n-gpu-layers".into(),
        total_layers.to_string(),
        "--split-mode".into(),
        "layer".into(),
        "--tensor-split".into(),
        layer_counts
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join("/"),
    ];
    if let Some(endpoint) = rpc {
        arguments.extend(["--rpc".into(), endpoint.into()]);
    }
    arguments
}

fn parse_benchmark_rates(output: &[u8]) -> Result<(f64, f64), ErrorPayload> {
    let rows: Vec<serde_json::Value> = serde_json::from_slice(output)
        .map_err(|error| ErrorPayload::new("benchmark_output", error.to_string(), None))?;
    let rate_for = |prompt: bool| {
        rows.iter().find_map(|row| {
            let n_prompt = row.get("n_prompt")?.as_u64()?;
            let n_gen = row.get("n_gen")?.as_u64()?;
            ((prompt && n_prompt > 0 && n_gen == 0) || (!prompt && n_prompt == 0 && n_gen > 0))
                .then(|| row.get("avg_ts")?.as_f64())
                .flatten()
        })
    };
    match (rate_for(true), rate_for(false)) {
        (Some(prompt), Some(generation)) => Ok((prompt, generation)),
        _ => Err(ErrorPayload::new(
            "benchmark_output",
            "llama-bench did not return separate prompt and generation measurements.",
            None,
        )),
    }
}

#[tauri::command]
pub fn cancel_inference_benchmark(state: State<'_, AppState>) -> Result<(), ErrorPayload> {
    if let Some(cancel) = state
        .benchmark_cancel
        .lock()
        .map_err(|_| benchmark_state_error())?
        .take()
    {
        let _ = cancel.send(());
    }
    Ok(())
}

fn benchmark_state_error() -> ErrorPayload {
    ErrorPayload::new(
        "benchmark_state",
        "Benchmark cancellation state is unavailable.",
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::{benchmark_arguments, parse_benchmark_rates};

    #[test]
    fn distributed_arguments_register_rpc_and_split_layers_across_both_gpus() {
        let arguments = benchmark_arguments("model.gguf", &[24, 16], Some("127.0.0.1:50052"));

        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--rpc", "127.0.0.1:50052"]));
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--n-gpu-layers", "40"]));
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--tensor-split", "24/16"]));
    }

    #[test]
    fn benchmark_output_keeps_prompt_and_generation_rates_separate() {
        let output = br#"[
            {"n_prompt":512,"n_gen":0,"avg_ts":220.5},
            {"n_prompt":0,"n_gen":128,"avg_ts":31.25}
        ]"#;

        assert_eq!(parse_benchmark_rates(output).unwrap(), (220.5, 31.25));
    }
}
