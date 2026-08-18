use std::{
    fs::{self, File},
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{atomic::{AtomicBool, Ordering}, Mutex},
    time::Duration,
};

use serde_json::Value;
use tauri::{path::BaseDirectory, AppHandle, Manager, State};

use crate::types::ErrorPayload;

const CONTROL_ORIGIN: &str = "http://127.0.0.1:11436";
const CONTROL_ADDRESS: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 11_436);

#[derive(Default)]
pub struct BackendProcess {
    child: Mutex<Option<Child>>,
    cluster_running: AtomicBool,
}

impl BackendProcess {
    pub fn start(&self, app: &AppHandle) -> Result<(), ErrorPayload> {
        if backend_is_listening() {
            return Ok(());
        }
        let mut child = self.child.lock().map_err(|_| bridge_error("Backend process state is unavailable."))?;
        if child.is_some() {
            return Ok(());
        }
        let (mut command, label) = backend_command(app)?;
        let log_path = logs_root().join("python-sidecar.log");
        fs::create_dir_all(log_path.parent().unwrap_or_else(|| Path::new(".")))
            .map_err(|error| bridge_error(error.to_string()))?;
        let stdout = File::create(&log_path).map_err(|error| bridge_error(error.to_string()))?;
        let stderr = stdout.try_clone().map_err(|error| bridge_error(error.to_string()))?;
        command
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        let process = command.spawn().map_err(|error| {
            ErrorPayload::new(
                "python_backend_start_failed",
                format!("Could not start {label}: {error}"),
                Some("Run `pnpm backend:install` and retry.".into()),
            )
        })?;
        *child = Some(process);
        Ok(())
    }

    pub fn is_cluster_running(&self) -> bool {
        self.cluster_running.load(Ordering::Relaxed)
    }

    fn record_command(&self, command: &str) {
        match command {
            "start_cluster" => self.cluster_running.store(true, Ordering::Relaxed),
            "stop_cluster" => self.cluster_running.store(false, Ordering::Relaxed),
            _ => {}
        }
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        if let Ok(slot) = self.child.get_mut() {
            if let Some(child) = slot.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[tauri::command]
pub async fn backend_request(
    command: String,
    args: Value,
    backend: State<'_, BackendProcess>,
) -> Result<Value, ErrorPayload> {
    let client = reqwest::Client::new();
    let url = format!("{CONTROL_ORIGIN}/_internal/{command}");
    let mut last_error = None;
    for _ in 0..60 {
        match client.post(&url).json(&args).send().await {
            Ok(response) => {
                let status = response.status();
                let value: Value = response.json().await.map_err(|error| {
                    ErrorPayload::new("python_backend_response", error.to_string(), None)
                })?;
                if !status.is_success() {
                    return Err(error_from_value(value));
                }
                if command == "update_settings" {
                    if let Some(enabled) = args
                        .get("settings")
                        .and_then(|settings| settings.get("autostart"))
                        .and_then(Value::as_bool)
                    {
                        crate::autostart::apply(enabled)?;
                    }
                }
                backend.record_command(&command);
                return Ok(value);
            }
            Err(error) if error.is_connect() => {
                last_error = Some(error.to_string());
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            Err(error) => {
                return Err(ErrorPayload::new("python_backend_request", error.to_string(), None));
            }
        }
    }
    Err(ErrorPayload::new(
        "python_backend_unavailable",
        last_error.unwrap_or_else(|| "The Python backend did not become ready.".into()),
        Some("Check the Python sidecar log in the SharedLocalLLM logs folder.".into()),
    ))
}

#[tauri::command]
pub fn pick_model_directory() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Choose a GGUF model directory")
        .pick_folder()
        .map(|path| path.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn open_network_settings() -> Result<(), ErrorPayload> {
    #[cfg(windows)]
    {
        Command::new("explorer.exe")
            .arg("ms-settings:network-status")
            .spawn()
            .map_err(|error| bridge_error(error.to_string()))?;
    }
    Ok(())
}

#[tauri::command]
pub fn open_logs_folder() -> Result<(), ErrorPayload> {
    let path = logs_root();
    fs::create_dir_all(&path).map_err(|error| bridge_error(error.to_string()))?;
    #[cfg(windows)]
    Command::new("explorer.exe")
        .arg(path)
        .spawn()
        .map_err(|error| bridge_error(error.to_string()))?;
    Ok(())
}

fn backend_command(app: &AppHandle) -> Result<(Command, String), ErrorPayload> {
    if let Ok(path) = std::env::var("SHARED_LOCAL_LLM_BACKEND_EXECUTABLE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok((Command::new(&path), path.display().to_string()));
        }
    }

    if let Ok(resource) = app.path().resolve(
        "backend/sharedlocalllm-backend.exe",
        BaseDirectory::Resource,
    ) {
        if resource.is_file() {
            return Ok((Command::new(&resource), resource.display().to_string()));
        }
    }

    let backend_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("backend");
    let configured = std::env::var("SHARED_LOCAL_LLM_BACKEND_PYTHON")
        .ok()
        .map(PathBuf::from);
    let venv = backend_dir.join(".venv").join("Scripts").join("python.exe");
    if let Some(python) = configured.filter(|path| path.is_file()).or_else(|| venv.is_file().then_some(venv)) {
        let mut command = Command::new(&python);
        command.args(["-m", "sharedlocalllm_backend"]).current_dir(&backend_dir);
        return Ok((command, python.display().to_string()));
    }

    Err(ErrorPayload::new(
        "python_backend_missing",
        "The SharedLocalLLM Python backend is not installed.",
        Some("Run `pnpm backend:install` from the repository root.".into()),
    ))
}

fn backend_is_listening() -> bool {
    TcpStream::connect_timeout(&CONTROL_ADDRESS, Duration::from_millis(100)).is_ok()
}

fn logs_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("SharedLocalLLM")
        .join("logs")
}

fn error_from_value(value: Value) -> ErrorPayload {
    ErrorPayload::new(
        value.get("code").and_then(Value::as_str).unwrap_or("python_backend_error"),
        value.get("message").and_then(Value::as_str).unwrap_or("The Python backend returned an error."),
        value.get("action").and_then(Value::as_str).map(str::to_owned),
    )
}

fn bridge_error(message: impl Into<String>) -> ErrorPayload {
    ErrorPayload::new("python_backend_bridge", message.into(), None)
}
