use std::{
    fs::{self, File},
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{ipc::Channel, path::BaseDirectory, AppHandle, Manager, State};

use crate::types::ErrorPayload;

const CONTROL_ORIGIN: &str = "http://127.0.0.1:11436";
const CONTROL_ADDRESS: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 11_436);

#[derive(Default)]
pub struct BackendProcess {
    child: Mutex<Option<Child>>,
    cluster_running: AtomicBool,
    watcher_stop: Arc<AtomicBool>,
    watcher: Mutex<Option<JoinHandle<()>>>,
}

impl BackendProcess {
    pub fn start(&self, app: &AppHandle) -> Result<(), ErrorPayload> {
        if backend_is_listening() {
            return Ok(());
        }
        let mut child = self
            .child
            .lock()
            .map_err(|_| bridge_error("Backend process state is unavailable."))?;
        if child.is_some() {
            return Ok(());
        }
        let process = spawn_backend(app)?;
        *child = Some(process);
        drop(child);
        self.spawn_dev_watcher(app);
        Ok(())
    }

    pub fn is_cluster_running(&self) -> bool {
        self.cluster_running.load(Ordering::Relaxed)
    }

    fn restart_backend(&self, app: &AppHandle) -> Result<(), ErrorPayload> {
        self.kill_backend()?;
        let process = spawn_backend(app)?;
        let mut slot = self
            .child
            .lock()
            .map_err(|_| bridge_error("Backend process state is unavailable."))?;
        *slot = Some(process);
        Ok(())
    }

    fn kill_backend(&self) -> Result<(), ErrorPayload> {
        let mut slot = self
            .child
            .lock()
            .map_err(|_| bridge_error("Backend process state is unavailable."))?;
        if let Some(child) = slot.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *slot = None;
        self.cluster_running.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// In dev builds, watch the Python sources and restart the backend when they change.
    fn spawn_dev_watcher(&self, app: &AppHandle) {
        if !cfg!(debug_assertions) {
            return;
        }
        if std::env::var("SHARED_LOCAL_LLM_BACKEND_RELOAD").as_deref() == Ok("0") {
            return;
        }
        // A custom executable has no local sources to reload from.
        if std::env::var("SHARED_LOCAL_LLM_BACKEND_EXECUTABLE").is_ok() {
            return;
        }
        let Some(backend_dir) = development_backend_dir() else {
            return;
        };
        if !backend_dir.join("sharedlocalllm_backend").is_dir() {
            return;
        }
        let mut slot = match self.watcher.lock() {
            Ok(slot) => slot,
            Err(_) => return,
        };
        if slot.is_some() {
            return;
        }
        let app = app.clone();
        let stop = Arc::clone(&self.watcher_stop);
        let handle = thread::spawn(move || {
            let mut signature = backend_signature(&backend_dir);
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(500));
                let next = backend_signature(&backend_dir);
                if next != signature {
                    signature = next;
                    // Let editors finish writing multi-file saves before restarting.
                    thread::sleep(Duration::from_millis(300));
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Some(backend) = app.try_state::<BackendProcess>() {
                        let _ = backend.restart_backend(&app);
                    }
                }
            }
        });
        *slot = Some(handle);
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
        self.watcher_stop.store(true, Ordering::Relaxed);
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
                return Err(ErrorPayload::new(
                    "python_backend_request",
                    error.to_string(),
                    None,
                ));
            }
        }
    }
    Err(ErrorPayload::new(
        "python_backend_unavailable",
        last_error.unwrap_or_else(|| "The Python backend did not become ready.".into()),
        Some("Check the Python sidecar log in the SharedLocalLLM logs folder.".into()),
    ))
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ChatStreamEvent {
    #[serde(rename_all = "camelCase")]
    Reasoning {
        content: String,
    },
    #[serde(rename_all = "camelCase")]
    Token {
        content: String,
    },
    #[serde(rename_all = "camelCase")]
    Stats {
        #[serde(rename = "tokensPerSecond")]
        tokens_per_second: f64,
    },
    Done,
    #[serde(rename_all = "camelCase")]
    Error {
        message: String,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatResponse {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
}

#[tauri::command]
pub async fn backend_stream(
    command: String,
    args: Value,
    channel: Channel<ChatStreamEvent>,
    backend: State<'_, BackendProcess>,
) -> Result<ChatResponse, ErrorPayload> {
    let _ = backend;
    let client = reqwest::Client::new();
    let url = format!("{CONTROL_ORIGIN}/_internal/stream/{command}");
    let response = client
        .post(&url)
        .json(&args)
        .send()
        .await
        .map_err(|error| ErrorPayload::new("python_backend_stream", error.to_string(), None))?;
    let status = response.status();
    if !status.is_success() {
        let value: Value = response.json().await.map_err(|error| {
            ErrorPayload::new("python_backend_response", error.to_string(), None)
        })?;
        return Err(error_from_value(value));
    }
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tokens_per_second = None;
    let mut buffer = String::new();
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|error| ErrorPayload::new("python_backend_stream", error.to_string(), None))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(position) = buffer.find("\n\n") {
            let event = buffer[..position].to_string();
            buffer = buffer[position + 2..].to_string();
            for line in event.lines() {
                let line = line.trim();
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let parsed: ChatStreamEvent = match serde_json::from_str(data.trim()) {
                    Ok(parsed) => parsed,
                    Err(_) => continue,
                };
                let _ = channel.send(parsed.clone());
                match &parsed {
                    ChatStreamEvent::Token { content: text } => content.push_str(text),
                    ChatStreamEvent::Reasoning { content: text } => reasoning.push_str(text),
                    ChatStreamEvent::Stats {
                        tokens_per_second: tps,
                    } => tokens_per_second = Some(*tps),
                    ChatStreamEvent::Done => {}
                    ChatStreamEvent::Error { message } => {
                        return Err(ErrorPayload::new(
                            "python_backend_stream",
                            message.clone(),
                            None,
                        ));
                    }
                }
            }
        }
    }
    Ok(ChatResponse {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        tokens_per_second,
    })
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

fn spawn_backend(app: &AppHandle) -> Result<Child, ErrorPayload> {
    let (mut command, label) = backend_command(app)?;
    command.env("PYTHONUNBUFFERED", "1").stdin(Stdio::null());
    if cfg!(debug_assertions) {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        let log_path = logs_root().join("python-sidecar.log");
        fs::create_dir_all(log_path.parent().unwrap_or_else(|| Path::new(".")))
            .map_err(|error| bridge_error(error.to_string()))?;
        let stdout = File::create(&log_path).map_err(|error| bridge_error(error.to_string()))?;
        let stderr = stdout
            .try_clone()
            .map_err(|error| bridge_error(error.to_string()))?;
        command
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
    }
    command.spawn().map_err(|error| {
        ErrorPayload::new(
            "python_backend_start_failed",
            format!("Could not start {label}: {error}"),
            Some("Run `pnpm backend:install` and retry.".into()),
        )
    })
}

fn backend_command(app: &AppHandle) -> Result<(Command, String), ErrorPayload> {
    if let Ok(path) = std::env::var("SHARED_LOCAL_LLM_BACKEND_EXECUTABLE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok((Command::new(&path), path.display().to_string()));
        }
    }

    if cfg!(debug_assertions) {
        if let Some(command) = development_backend_command() {
            return Ok(command);
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

    if let Some(command) = development_backend_command() {
        return Ok(command);
    }

    Err(ErrorPayload::new(
        "python_backend_missing",
        "The SharedLocalLLM Python backend is not installed.",
        Some("Run `pnpm backend:install` from the repository root.".into()),
    ))
}

fn development_backend_dir() -> Option<PathBuf> {
    let backend_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("backend");
    backend_dir.is_dir().then_some(backend_dir)
}

fn development_backend_command() -> Option<(Command, String)> {
    let backend_dir = development_backend_dir()?;
    let configured = std::env::var("SHARED_LOCAL_LLM_BACKEND_PYTHON")
        .ok()
        .map(PathBuf::from);
    let venv = backend_dir.join(".venv").join("Scripts").join("python.exe");
    let python = configured
        .filter(|path| path.is_file())
        .or_else(|| venv.is_file().then_some(venv))?;
    let mut command = Command::new(&python);
    command
        .args(["-m", "sharedlocalllm_backend"])
        .current_dir(&backend_dir);
    Some((command, python.display().to_string()))
}

/// Snapshot of the tracked backend sources: (path, mtime, size).
type FileSignature = Vec<(PathBuf, u64, u64)>;

/// Fingerprint of the Python sources that drive the running backend. Only the
/// package directory, the sidecar entry point, and pyproject.toml are tracked,
/// so venv/installation churn never triggers a dev reload.
fn backend_signature(backend_dir: &Path) -> FileSignature {
    let mut entries = Vec::new();
    let package = backend_dir.join("sharedlocalllm_backend");
    if package.is_dir() {
        collect_py_files(&package, &mut entries);
    }
    for name in ["sidecar_entry.py", "pyproject.toml"] {
        if let Some(entry) = file_signature(&backend_dir.join(name)) {
            entries.push(entry);
        }
    }
    entries.sort();
    entries
}

fn collect_py_files(dir: &Path, out: &mut FileSignature) {
    let Ok(reader) = fs::read_dir(dir) else {
        return;
    };
    for entry in reader.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_py_files(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("py") {
            if let Some(signature) = file_signature(&path) {
                out.push(signature);
            }
        }
    }
}

fn file_signature(path: &Path) -> Option<(PathBuf, u64, u64)> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let since = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some((
        path.to_path_buf(),
        since.as_secs(),
        since.subsec_nanos() as u64,
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
        value
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or("python_backend_error"),
        value
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("The Python backend returned an error."),
        value
            .get("action")
            .and_then(Value::as_str)
            .map(str::to_owned),
    )
}

fn bridge_error(message: impl Into<String>) -> ErrorPayload {
    ErrorPayload::new("python_backend_bridge", message.into(), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let unique = format!(
                "SharedLocalLLM-watcher-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_nanos())
                    .unwrap_or(0)
            );
            let path = std::env::temp_dir().join(unique);
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_touched(path: &Path, contents: &str, seconds: u64) {
        fs::write(path, contents).expect("write temp file");
        let file = fs::File::options()
            .write(true)
            .open(path)
            .expect("open temp file");
        file.set_modified(UNIX_EPOCH + Duration::from_secs(seconds))
            .expect("set modified time");
    }

    #[test]
    fn backend_signature_detects_edited_python_file() {
        let dir = TempDir::new();
        let package = dir.path().join("sharedlocalllm_backend");
        fs::create_dir_all(&package).unwrap();
        let module = package.join("runtime.py");
        write_touched(&module, "print('one')", 1000);
        let before = backend_signature(dir.path());
        write_touched(&module, "print('two')", 2000);
        assert_ne!(before, backend_signature(dir.path()));
    }

    #[test]
    fn backend_signature_ignores_untracked_files() {
        let dir = TempDir::new();
        let package = dir.path().join("sharedlocalllm_backend");
        fs::create_dir_all(&package).unwrap();
        write_touched(&package.join("runtime.py"), "print('one')", 1000);
        let before = backend_signature(dir.path());

        let venv = dir.path().join(".venv").join("Scripts");
        fs::create_dir_all(&venv).unwrap();
        write_touched(&venv.join("site.py"), "print('venv')", 3000);
        write_touched(&dir.path().join("unused.py"), "print('untracked')", 4000);
        fs::write(dir.path().join("README.md"), "# untracked").expect("write README");

        assert_eq!(before, backend_signature(dir.path()));
    }

    #[test]
    fn backend_signature_tracks_pyproject_toml() {
        let dir = TempDir::new();
        let package = dir.path().join("sharedlocalllm_backend");
        fs::create_dir_all(&package).unwrap();
        write_touched(&package.join("runtime.py"), "print('x')", 1000);
        let manifest = dir.path().join("pyproject.toml");
        write_touched(&manifest, "dependencies = []", 1000);
        let before = backend_signature(dir.path());
        write_touched(&manifest, "dependencies = ['fastapi']", 2000);
        assert_ne!(before, backend_signature(dir.path()));
    }
}
