use std::process::{Child, Command, Stdio};

use super::runtime_root;
use crate::types::{ErrorPayload, ModelRecord};

#[derive(Default)]
pub struct ProcessManager {
    pub server: Option<Child>,
    pub rpc: Option<Child>,
}

impl ProcessManager {
    pub fn start(
        &mut self,
        model: &ModelRecord,
        context: u32,
        api_key: &str,
        use_rpc: bool,
        rpc_endpoint: Option<String>,
        api_port: u16,
    ) -> Result<(), ErrorPayload> {
        self.stop();
        let root = runtime_root().join("current");
        for name in ["llama-server.exe"] {
            if !root.join(name).is_file() {
                return Err(ErrorPayload::new(
                    "runtime_missing",
                    format!("{name} is not installed."),
                    Some("Install the pinned llama.cpp runtime from Setup.".into()),
                ));
            }
        }
        let context = context.to_string();
        let api_port = api_port.to_string();
        let mut command = Command::new(root.join("llama-server.exe"));
        command.args([
            "--host",
            "127.0.0.1",
            "--port",
            &api_port,
            "--model",
            &model.shard_paths[0],
            "--ctx-size",
            &context,
            "--n-gpu-layers",
            "999",
            "--fit",
            "on",
            "--split-mode",
            "layer",
            "--api-key",
            api_key,
        ]);
        if use_rpc {
            let endpoint = rpc_endpoint.ok_or_else(|| {
                ErrorPayload::new(
                    "rpc_tunnel_missing",
                    "The encrypted RPC forwarder is unavailable.",
                    None,
                )
            })?;
            command.args(["--rpc", &endpoint]);
        }
        if let Some(projector) = &model.projector {
            command.args(["--mmproj", projector]);
        }
        self.server = Some(
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(process_error)?,
        );
        Ok(())
    }

    pub fn stop(&mut self) {
        terminate(&mut self.server);
        terminate(&mut self.rpc);
    }
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
