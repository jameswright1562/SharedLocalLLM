use std::{
    fs::{self, File},
    process::{Child, Command, Stdio},
};

use super::{runtime_root, ProcessJob};
use crate::types::{ErrorPayload, ModelLoadConfig, ModelRecord};

#[derive(Default)]
pub struct ProcessManager {
    pub server: Option<Child>,
    pub rpc: Option<Child>,
    job: Option<ProcessJob>,
}

impl ProcessManager {
    pub fn start(
        &mut self,
        model: &ModelRecord,
        load_config: &ModelLoadConfig,
        api_key: &str,
        use_rpc: bool,
        rpc_endpoint: Option<String>,
        api_port: u16,
    ) -> Result<(), ErrorPayload> {
        self.stop();
        let root = runtime_root().join("current");
        if !root.join("llama-server.exe").is_file() {
            return Err(ErrorPayload::new(
                "runtime_missing",
                "llama-server.exe is not installed.",
                Some("Install the pinned llama.cpp runtime from Setup.".into()),
            ));
        }
        let mut command = Command::new(root.join("llama-server.exe"));
        let rpc_endpoint = if use_rpc {
            Some(rpc_endpoint.as_deref().ok_or_else(|| {
                ErrorPayload::new(
                    "rpc_tunnel_missing",
                    "The encrypted RPC forwarder is unavailable.",
                    None,
                )
            })?)
        } else {
            None
        };
        let layer_counts = load_config
            .gpu_layers
            .iter()
            .map(|allocation| allocation.layers)
            .collect::<Vec<_>>();
        let gpu_layer_count = layer_counts.iter().copied().sum();
        command.args(server_arguments(ServerArguments {
            model_path: &model.shard_paths[0],
            context: load_config.context_size.max(4096),
            api_port,
            gpu_layer_count,
            tensor_split: &layer_counts,
            rpc_endpoint,
            projector: model.projector.as_deref(),
        }));
        command.env("LLAMA_ARG_API_KEY", api_key);
        command.env("LLAMA_API_KEY", api_key);
        let log_path = dirs::data_local_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("SharedLocalLLM")
            .join("logs")
            .join("llama-server.stderr.log");
        if let Some(parent) = log_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let stderr = File::create(&log_path).map_err(process_error)?;
        let job = ProcessJob::new();
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(process_error)?;
        if let Some(job) = job.as_ref() {
            job.assign(&child)?;
        }
        self.job = job;
        self.server = Some(child);
        Ok(())
    }

    pub fn stop(&mut self) {
        terminate(&mut self.server);
        terminate(&mut self.rpc);
        self.job = None;
    }
}

struct ServerArguments<'a> {
    model_path: &'a str,
    context: u32,
    api_port: u16,
    gpu_layer_count: u32,
    tensor_split: &'a [u32],
    rpc_endpoint: Option<&'a str>,
    projector: Option<&'a str>,
}

fn server_arguments(config: ServerArguments<'_>) -> Vec<String> {
    let mut arguments = vec![
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        config.api_port.to_string(),
        "--model".into(),
        config.model_path.into(),
        "--ctx-size".into(),
        config.context.to_string(),
        "--n-gpu-layers".into(),
        if config.tensor_split.is_empty() {
            "auto".into()
        } else {
            config.gpu_layer_count.to_string()
        },
        "--fit".into(),
        if config.tensor_split.is_empty() {
            "on".into()
        } else {
            "off".into()
        },
        "--split-mode".into(),
        "layer".into(),
    ];
    if !config.tensor_split.is_empty() {
        arguments.extend([
            "--tensor-split".into(),
            config
                .tensor_split
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ]);
    }
    if let Some(endpoint) = config.rpc_endpoint {
        arguments.extend(["--rpc".into(), endpoint.into()]);
    }
    if let Some(projector) = config.projector {
        arguments.extend(["--mmproj".into(), projector.into()]);
    }
    arguments
}

impl Drop for ProcessManager {
    fn drop(&mut self) {
        self.stop();
    }
}

fn terminate(child: &mut Option<Child>) {
    if let Some(mut process) = child.take() {
        let _ = process.kill();
        let _ = process.wait();
    }
}

fn process_error(error: std::io::Error) -> ErrorPayload {
    ErrorPayload::new(
        "runtime_process_failed",
        error.to_string(),
        Some("Inspect the local runtime logs and verify the pinned files.".into()),
    )
}

#[cfg(test)]
mod tests {
    use super::{server_arguments, ServerArguments};

    #[test]
    fn manual_layer_split_is_forwarded_to_llama_server() {
        let arguments = server_arguments(ServerArguments {
            model_path: "model.gguf",
            context: 8_192,
            api_port: 11_435,
            gpu_layer_count: 40,
            tensor_split: &[24, 16],
            rpc_endpoint: Some("127.0.0.1:50052"),
            projector: None,
        });

        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--n-gpu-layers", "40"]));
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--tensor-split", "24,16"]));
        assert!(arguments.windows(2).any(|pair| pair == ["--fit", "off"]));
        assert!(arguments
            .windows(2)
            .any(|pair| pair == ["--rpc", "127.0.0.1:50052"]));
        assert!(!arguments.iter().any(|argument| argument == "--api-key"));
    }
}
