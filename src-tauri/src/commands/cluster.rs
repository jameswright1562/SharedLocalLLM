use std::time::Duration;

use tauri::State;

use crate::{
    runtime,
    state::AppState,
    types::{ClusterSession, ErrorPayload, InferenceBenchmark},
};

#[tauri::command]
pub async fn start_cluster(
    model_id: String,
    context_size: u32,
    state: State<'_, AppState>,
) -> Result<ClusterSession, ErrorPayload> {
    let (model, api_key, api_port, coordinator, peer_id) = {
        let inner = state.lock()?;
        let model = inner
            .models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
            .ok_or_else(|| {
                ErrorPayload::new(
                    "model_not_found",
                    "Refresh the catalogue and choose an available model.",
                    None,
                )
            })?;
        (
            model,
            inner.api_key.clone(),
            inner.api_port,
            inner.local.id.clone(),
            inner.peers.first().map(|peer| peer.id.clone()),
        )
    };
    if runtime::status().status != "ready" {
        return Err(ErrorPayload::new(
            "runtime_missing",
            "The pinned llama.cpp runtime is not ready.",
            Some("Install it from the first-run setup.".into()),
        ));
    }
    let rpc_endpoint = if peer_id.is_some() {
        let client = state.peer_client().await?;
        client.heartbeat().await?;
        let forwarder = client.start_rpc_forwarder().await?;
        let endpoint = forwarder.local_address().to_string();
        state.peer.lock().await.forwarder = Some(forwarder);
        Some(endpoint)
    } else {
        None
    };
    state
        .processes
        .lock()
        .map_err(|_| {
            ErrorPayload::new(
                "process_state",
                "The runtime process manager is unavailable.",
                None,
            )
        })?
        .start(
            &model,
            context_size.max(4096),
            &api_key,
            peer_id.is_some(),
            rpc_endpoint,
            api_port,
        )?;
    if let Err(error) = wait_for_health(api_port, &api_key).await {
        state
            .processes
            .lock()
            .map_err(|_| {
                ErrorPayload::new(
                    "process_state",
                    "The runtime process manager is unavailable.",
                    None,
                )
            })?
            .stop();
        if let Some(forwarder) = state.peer.lock().await.forwarder.take() {
            forwarder.shutdown().await;
        }
        return Err(error);
    }
    let session = ClusterSession {
        status: "running".into(),
        coordinator_node_id: Some(coordinator),
        worker_node_id: peer_id,
        model_id: Some(model_id),
        error: None,
    };
    state.lock()?.cluster = session.clone();
    Ok(session)
}

#[tauri::command]
pub async fn stop_cluster(state: State<'_, AppState>) -> Result<ClusterSession, ErrorPayload> {
    state
        .processes
        .lock()
        .map_err(|_| {
            ErrorPayload::new(
                "process_state",
                "The runtime process manager is unavailable.",
                None,
            )
        })?
        .stop();
    if let Some(forwarder) = state.peer.lock().await.forwarder.take() {
        forwarder.shutdown().await;
    }
    let mut inner = state.lock()?;
    inner.cluster = ClusterSession {
        status: if inner.peers.is_empty() {
            "idle"
        } else {
            "ready"
        }
        .into(),
        ..ClusterSession::default()
    };
    Ok(inner.cluster.clone())
}

#[tauri::command]
pub async fn run_inference_benchmark(
    model_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<InferenceBenchmark>, ErrorPayload> {
    let model = state
        .lock()?
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
    let mut command = tokio::process::Command::new(executable);
    command
        .args([
            "-m",
            &model.shard_paths[0],
            "-p",
            "512",
            "-n",
            "128",
            "-r",
            "3",
            "-o",
            "json",
        ])
        .kill_on_drop(true);
    let (cancel, cancelled) = tokio::sync::oneshot::channel();
    *state.benchmark_cancel.lock().map_err(|_| {
        ErrorPayload::new(
            "benchmark_state",
            "Benchmark cancellation state is unavailable.",
            None,
        )
    })? = Some(cancel);
    let output = tokio::select! {
        output = command.output() => output.map_err(|error| ErrorPayload::new("benchmark_process", error.to_string(), None))?,
        _ = cancelled => return Err(ErrorPayload::new("benchmark_cancelled", "The inference benchmark was cancelled.", None)),
    };
    let _ = state.benchmark_cancel.lock().map(|mut slot| slot.take());
    if !output.status.success() {
        return Err(ErrorPayload::new(
            "benchmark_failed",
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            Some("Review the model fit recommendation and runtime logs.".into()),
        ));
    }
    let raw: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| ErrorPayload::new("benchmark_output", error.to_string(), None))?;
    let rows = raw.as_array().cloned().unwrap_or_default();
    let generation = rows
        .iter()
        .find_map(|row| row.get("avg_ts").and_then(|v| v.as_f64()))
        .unwrap_or_default();
    let result = InferenceBenchmark {
        id: uuid::Uuid::new_v4().to_string(),
        model_name: model.name,
        topology: "local".into(),
        prompt_tokens_per_second: generation,
        generation_tokens_per_second: generation,
        load_time_seconds: 0.0,
        memory_peak_gb: 0.0,
        recommended: true,
        ran_at: format!("{}", crate::pairing::now()),
    };
    state.lock()?.benchmarks.push(result.clone());
    state.persist()?;
    Ok(vec![result])
}

#[tauri::command]
pub fn cancel_inference_benchmark(state: State<'_, AppState>) -> Result<(), ErrorPayload> {
    if let Some(cancel) = state
        .benchmark_cancel
        .lock()
        .map_err(|_| {
            ErrorPayload::new(
                "benchmark_state",
                "Benchmark cancellation state is unavailable.",
                None,
            )
        })?
        .take()
    {
        let _ = cancel.send(());
    }
    Ok(())
}

async fn wait_for_health(port: u16, api_key: &str) -> Result<(), ErrorPayload> {
    let url = format!("http://127.0.0.1:{port}/health");
    for _ in 0..40 {
        if reqwest::Client::new()
            .get(&url)
            .bearer_auth(api_key)
            .timeout(Duration::from_millis(500))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Err(ErrorPayload::new(
        "llama_server_not_ready",
        "llama-server did not become healthy within ten seconds.",
        Some("Open the logs folder and check model/runtime compatibility.".into()),
    ))
}
