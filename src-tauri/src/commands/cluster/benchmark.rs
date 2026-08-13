use tauri::State;

use crate::{
    runtime,
    state::AppState,
    types::{ErrorPayload, InferenceBenchmark},
};

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
    *state
        .benchmark_cancel
        .lock()
        .map_err(|_| benchmark_state_error())? = Some(cancel);
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
    let generation = raw
        .as_array()
        .and_then(|rows| {
            rows.iter()
                .find_map(|row| row.get("avg_ts").and_then(|value| value.as_f64()))
        })
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
