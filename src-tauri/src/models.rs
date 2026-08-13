use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    process::Command,
};

use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::{
    gguf::read_model_metadata,
    types::{ErrorPayload, ModelLocation, ModelRecord},
};

pub fn default_lm_studio_root() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".lmstudio").join("models"))
}

pub fn lms_catalog_roots() -> Vec<PathBuf> {
    let Ok(output) = Command::new("lms")
        .args(["ls", "--json", "--detailed"])
        .output()
    else {
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
mod default_root_tests {
    use super::{discover_gguf_models, expand_lm_studio_roots, lm_studio_roots_for_home};

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

        let defaults = lm_studio_roots_for_home(home.path());
        assert!(defaults.contains(&legacy));
        assert!(defaults.contains(&hub));
        assert!(defaults.contains(&downloads));

        let manual_roots = expand_lm_studio_roots(&[hub]);
        assert!(manual_roots.contains(&downloads));
        assert_eq!(discover_gguf_models(&manual_roots).unwrap().len(), 1);
    }
}

pub fn discover_gguf_models(roots: &[PathBuf]) -> Result<Vec<ModelRecord>, ErrorPayload> {
    let split = Regex::new(r"(?i)^(.*)-(\d{5})-of-(\d{5})\.gguf$").expect("valid split regex");
    let quant = Regex::new(r"(?i)(Q\d(?:_[A-Z0-9]+)+|F16|F32|BF16)").expect("valid quant regex");
    let mut files = HashSet::new();
    for root in roots {
        if !root.exists() {
            continue;
        }
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if entry.file_type().is_file()
                && path
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
            {
                let valid_magic = std::fs::File::open(path)
                    .ok()
                    .and_then(|mut file| {
                        use std::io::Read;
                        let mut magic = [0u8; 4];
                        file.read_exact(&mut magic).ok().map(|_| magic == *b"GGUF")
                    })
                    .unwrap_or(false);
                if !valid_magic {
                    continue;
                }
                if let Ok(canonical) = path.canonicalize() {
                    files.insert(canonical);
                }
            }
        }
    }

    let projectors: Vec<PathBuf> = files
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.to_ascii_lowercase().starts_with("mmproj"))
        })
        .cloned()
        .collect();
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for path in files.into_iter().filter(|path| !projectors.contains(path)) {
        let filename = path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or_default();
        let stem = split
            .captures(filename)
            .and_then(|captures| captures.get(1))
            .map(|x| x.as_str().to_owned())
            .unwrap_or_else(|| {
                filename
                    .strip_suffix(".gguf")
                    .or_else(|| filename.strip_suffix(".GGUF"))
                    .unwrap_or(filename)
                    .to_owned()
            });
        let key = format!(
            "{}|{}",
            path.parent().unwrap_or(Path::new("")).display(),
            stem.to_ascii_lowercase()
        );
        groups.entry(key).or_default().push(path);
    }

    let mut records = Vec::new();
    for (key, mut shards) in groups {
        shards.sort();
        if shards.len() > 1 {
            let mut indices = Vec::new();
            let mut expected = None;
            for shard in &shards {
                let filename = shard
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();
                let Some(captures) = split.captures(filename) else {
                    continue;
                };
                indices.push(captures[2].parse::<usize>().unwrap_or_default());
                expected = captures[3].parse::<usize>().ok();
            }
            indices.sort_unstable();
            if expected != Some(shards.len()) || indices != (1..=shards.len()).collect::<Vec<_>>() {
                continue;
            }
        }
        let first = &shards[0];
        let metadata = read_model_metadata(first).unwrap_or_default();
        let base = key.rsplit('|').next().unwrap_or("model");
        let projector = projectors
            .iter()
            .find(|projector| projector.parent() == first.parent())
            .cloned();
        let size_bytes = shards
            .iter()
            .filter_map(|path| path.metadata().ok().map(|m| m.len()))
            .sum();
        let display_name = base.replace(['_', '-'], " ");
        let quantization = quant
            .find(base)
            .map(|m| m.as_str().to_ascii_uppercase())
            .unwrap_or_else(|| "Unknown".into());
        let lowered = first.to_string_lossy().to_ascii_lowercase();
        let source = if lowered.contains(".lmstudio") {
            "lm-studio"
        } else {
            "custom"
        };
        let mut hasher = Sha256::new();
        for path in &shards {
            hasher.update(path.to_string_lossy().as_bytes());
            if let Ok(metadata) = path.metadata() {
                hasher.update(metadata.len().to_le_bytes());
                if let Ok(modified) = metadata.modified().and_then(|value| {
                    value
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(std::io::Error::other)
                }) {
                    hasher.update(modified.as_nanos().to_le_bytes());
                }
            }
        }
        let shard_paths: Vec<String> = shards
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
        let location = ModelLocation {
            node_id: "local-node".into(),
            path: shard_paths[0].clone(),
            source: source.into(),
        };
        let vision_capable = projector.is_some();
        records.push(ModelRecord {
            id: hex::encode(hasher.finalize())[..16].to_owned(),
            name: title_case(&display_name),
            architecture: metadata
                .architecture
                .clone()
                .unwrap_or_else(|| infer_architecture(&lowered)),
            quantization,
            size_bytes,
            context_length: metadata.context_length.unwrap_or(8192),
            layer_count: metadata.block_count,
            embedding_length: metadata.embedding_length,
            attention_head_count: metadata.attention_head_count,
            attention_head_count_kv: metadata.attention_head_count_kv,
            capability: if vision_capable {
                "vision".into()
            } else {
                "text".into()
            },
            shards: shard_paths.len(),
            locations: vec![location],
            fit: "does-not-fit".into(),
            shard_paths,
            projector: projector.map(|path| path.to_string_lossy().into_owned()),
            vision_capable,
        });
    }
    Ok(records)
}

fn infer_architecture(path: &str) -> String {
    ["qwen", "llama", "mistral", "gemma", "phi", "deepseek"]
        .into_iter()
        .find(|name| path.contains(name))
        .unwrap_or("unknown")
        .into()
}

fn title_case(value: &str) -> String {
    value
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}
