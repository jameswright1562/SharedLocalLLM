use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use crate::types::ErrorPayload;

pub(super) fn flatten_required(root: &Path, required: &[String]) -> Result<(), ErrorPayload> {
    for name in required {
        if root.join(name).exists() {
            if let Some(parent) = root.parent() {
                copy_sibling_dlls(parent, root)?;
            }
            continue;
        }
        let found = walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .find(|entry| {
                entry.file_type().is_file()
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .eq_ignore_ascii_case(name)
            });
        if let Some(found) = found {
            fs::copy(found.path(), root.join(name)).map_err(runtime_io)?;
            if let Some(parent) = found.path().parent() {
                copy_sibling_dlls(parent, root)?;
            }
        }
    }
    Ok(())
}

fn copy_sibling_dlls(from: &Path, to: &Path) -> Result<(), ErrorPayload> {
    let entries = match fs::read_dir(from) {
        Ok(entries) => entries,
        Err(_) => return Ok(()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("dll"))
        {
            let dest = to.join(path.file_name().unwrap_or_default());
            if !dest.exists() {
                fs::copy(&path, dest).map_err(runtime_io)?;
            }
        }
    }
    Ok(())
}

pub(super) fn validate_required(root: &Path, required: &[String]) -> Result<(), ErrorPayload> {
    for executable in required {
        if !root.join(executable).is_file() {
            return Err(ErrorPayload::new(
                "runtime_incomplete",
                format!("Verified archives did not contain {executable}."),
                Some("Retry the download or import the matching official archive.".into()),
            ));
        }
        health_check(&root.join(executable))?;
    }
    Ok(())
}

fn health_check(executable: &Path) -> Result<(), ErrorPayload> {
    let mut child = Command::new(executable)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            ErrorPayload::new(
                "runtime_health_failed",
                error.to_string(),
                Some(
                    "The downloaded runtime may be incompatible with this Windows installation."
                        .into(),
                ),
            )
        })?;
    let started = Instant::now();
    loop {
        match child.try_wait().map_err(runtime_io)? {
            Some(status) if status.success() => return Ok(()),
            Some(_) => {
                return Err(ErrorPayload::new(
                    "runtime_health_failed",
                    format!(
                        "{} exited unsuccessfully from --version.",
                        executable.display()
                    ),
                    Some("The runtime may be missing companion DLLs.".into()),
                ));
            }
            None if started.elapsed() > Duration::from_secs(5) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ErrorPayload::new(
                    "runtime_health_timeout",
                    format!("{} did not answer --version.", executable.display()),
                    None,
                ));
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

pub(super) fn runtime_io(error: std::io::Error) -> ErrorPayload {
    ErrorPayload::new(
        "runtime_io",
        error.to_string(),
        Some("Check available disk space and folder permissions.".into()),
    )
}

pub(super) fn activate(root: &Path, extracted: &Path, version: &str) -> Result<(), ErrorPayload> {
    let current = root.join("current");
    let previous = root.join("previous");
    if previous.exists() {
        fs::remove_dir_all(&previous).map_err(runtime_io)?;
    }
    if current.exists() {
        fs::rename(&current, &previous).map_err(runtime_io)?;
    }
    if let Err(error) = fs::rename(extracted, &current) {
        if previous.exists() {
            let _ = fs::rename(&previous, &current);
        }
        return Err(runtime_io(error));
    }
    if let Err(error) = fs::write(current.join("version.txt"), version) {
        let _ = fs::remove_dir_all(&current);
        if previous.exists() {
            let _ = fs::rename(&previous, &current);
        }
        return Err(runtime_io(error));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::flatten_required;

    #[test]
    fn flatten_copies_sibling_dlls_with_nested_executables() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("bin");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("llama-server.exe"), b"exe").unwrap();
        std::fs::write(nested.join("ggml.dll"), b"dll").unwrap();
        flatten_required(root.path(), &["llama-server.exe".into()]).unwrap();
        assert!(root.path().join("llama-server.exe").is_file());
        assert!(root.path().join("ggml.dll").is_file());
    }
}
