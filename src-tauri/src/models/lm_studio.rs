use std::{
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::Value;

pub fn default_roots() -> Vec<PathBuf> {
    dirs::home_dir()
        .map(|home| roots_for_home(&home))
        .unwrap_or_default()
}

pub fn expand_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut expanded = roots.to_vec();
    for root in roots {
        let Some(lm_studio) = root
            .ancestors()
            .find(|ancestor| ancestor.file_name().is_some_and(|name| name == ".lmstudio"))
        else {
            continue;
        };
        if let Some(downloads) = configured_download_folder(lm_studio) {
            expanded.push(downloads);
        }
    }
    expanded.sort();
    expanded.dedup();
    expanded
}

fn roots_for_home(home: &Path) -> Vec<PathBuf> {
    let lm_studio = home.join(".lmstudio");
    let mut roots = vec![
        lm_studio.join("models"),
        lm_studio.join("hub").join("models"),
        lm_studio.join(".internal").join("bundled-models"),
    ];
    if let Some(downloads) = configured_download_folder(&lm_studio) {
        roots.push(downloads);
    }
    roots.retain(|path| path.is_dir());
    roots.sort();
    roots.dedup();
    roots
}

fn configured_download_folder(lm_studio: &Path) -> Option<PathBuf> {
    let settings = std::fs::read(lm_studio.join("settings.json")).ok()?;
    let value: Value = serde_json::from_slice(&settings).ok()?;
    let downloads = PathBuf::from(value.get("downloadsFolder")?.as_str()?);
    let downloads = if downloads.is_absolute() {
        downloads
    } else {
        lm_studio.join(downloads)
    };
    downloads.is_dir().then_some(downloads)
}

pub fn lms_catalog_roots() -> Vec<PathBuf> {
    let executable = dirs::home_dir()
        .map(|home| home.join(".lmstudio").join("bin").join("lms.exe"))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from("lms"));
    let Ok(output) = Command::new(executable).args(["ls", "--json"]).output() else {
        return vec![];
    };
    if !output.status.success() {
        return vec![];
    }
    let Ok(value) = serde_json::from_slice::<Value>(&output.stdout) else {
        return vec![];
    };
    let mut values = Vec::new();
    collect_paths(&value, &mut values);
    values
        .into_iter()
        .filter_map(|path| {
            let path = PathBuf::from(path);
            if path
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
            {
                path.parent().map(Path::to_path_buf)
            } else if path.is_dir() {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

fn collect_paths(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(value) if value.to_ascii_lowercase().contains(".gguf") => {
            output.push(value.clone())
        }
        Value::Array(values) => values.iter().for_each(|value| collect_paths(value, output)),
        Value::Object(values) => values
            .values()
            .for_each(|value| collect_paths(value, output)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_roots, roots_for_home};
    use crate::models::discover_gguf_models;

    #[test]
    fn resolves_hub_metadata_to_the_configured_lm_studio_download_folder() {
        let home = tempfile::tempdir().unwrap();
        let legacy = home.path().join(".lmstudio").join("models");
        let hub = home.path().join(".lmstudio").join("hub").join("models");
        let downloads = home.path().join("download-cache");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::create_dir_all(&hub).unwrap();
        let model_folder = downloads.join("publisher").join("model");
        std::fs::create_dir_all(&model_folder).unwrap();
        std::fs::write(model_folder.join("model-Q4_K_M.gguf"), b"GGUF").unwrap();
        std::fs::write(
            home.path().join(".lmstudio").join("settings.json"),
            serde_json::json!({ "downloadsFolder": downloads }).to_string(),
        )
        .unwrap();

        let defaults = roots_for_home(home.path());
        assert!(defaults.contains(&legacy));
        assert!(defaults.contains(&hub));
        assert!(defaults.contains(&downloads));
        let manual_roots = expand_roots(&[hub]);
        assert!(manual_roots.contains(&downloads));
        assert_eq!(discover_gguf_models(&manual_roots).unwrap().len(), 1);
    }
}
