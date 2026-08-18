use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    time::Duration,
};

use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use super::{
    manifest::{manifest, runtime_root, status, RuntimeAsset},
    verify::{activate, flatten_required, runtime_io, validate_required},
};
use crate::types::{ErrorPayload, RuntimeStatus};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeProgress {
    pub percent: u8,
    pub status: String,
}

pub async fn install<F>(mut progress: F) -> Result<RuntimeStatus, ErrorPayload>
where
    F: FnMut(RuntimeProgress) + Send,
{
    let manifest = manifest()?;
    if !manifest.enabled {
        return Err(ErrorPayload::new(
            "runtime_disabled",
            "Runtime installation is disabled by the bundled manifest.",
            None,
        ));
    }
    let release = manifest.release.ok_or_else(|| {
        ErrorPayload::new(
            "runtime_unpinned",
            "No pinned runtime release is configured.",
            None,
        )
    })?;
    let root = runtime_root();
    fs::create_dir_all(&root).map_err(runtime_io)?;
    let staging = root.join(format!("staging-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&staging).map_err(runtime_io)?;

    let result = async {
        emit(&mut progress, 2, "Downloading llama.cpp");
        let llama = download_verified(
            &release.llama_cpp,
            &staging.join("llama.zip"),
            2,
            42,
            &mut progress,
        )
        .await?;
        emit(&mut progress, 44, "Downloading CUDA runtime");
        let cuda = download_verified(
            &release.cuda_runtime,
            &staging.join("cuda.zip"),
            44,
            82,
            &mut progress,
        )
        .await?;
        let extracted = staging.join("extracted");
        fs::create_dir_all(&extracted).map_err(runtime_io)?;
        emit(&mut progress, 84, "Extracting verified archives");
        extract_archive(&llama, &extracted)?;
        extract_archive(&cuda, &extracted)?;
        flatten_required(&extracted, &manifest.required_executables)?;
        validate_required(&extracted, &manifest.required_executables)?;
        activate(&root, &extracted, &release.tag)?;
        emit(&mut progress, 100, "Runtime ready");
        Ok(status())
    }
    .await;
    let _ = fs::remove_dir_all(&staging);
    result
}

fn emit<F: FnMut(RuntimeProgress)>(progress: &mut F, percent: u8, status: &str) {
    progress(RuntimeProgress {
        percent,
        status: status.into(),
    });
}

async fn download_verified<F>(
    asset: &RuntimeAsset,
    path: &Path,
    start: u8,
    end: u8,
    progress: &mut F,
) -> Result<PathBuf, ErrorPayload>
where
    F: FnMut(RuntimeProgress),
{
    if !asset
        .url
        .starts_with("https://github.com/ggml-org/llama.cpp/releases/download/")
    {
        return Err(ErrorPayload::new(
            "runtime_origin_rejected",
            "The runtime asset is not hosted on the official llama.cpp GitHub release path.",
            None,
        ));
    }
    let response = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(30 * 60))
        .build()
        .map_err(download_error)?
        .get(&asset.url)
        .send()
        .await
        .map_err(download_error)?
        .error_for_status()
        .map_err(download_error)?;
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(path).await.map_err(runtime_io)?;
    let mut digest = Sha256::new();
    let mut count = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(download_error)?;
        count += chunk.len() as u64;
        if count > asset.size {
            return Err(ErrorPayload::new(
                "runtime_size_mismatch",
                "The runtime download exceeded its pinned byte size.",
                Some("The unexpected archive was rejected before activation.".into()),
            ));
        }
        digest.update(&chunk);
        file.write_all(&chunk).await.map_err(runtime_io)?;
        let ratio = (count as f64 / asset.size.max(1) as f64).min(1.0);
        emit(
            progress,
            start + ((end - start) as f64 * ratio) as u8,
            "Downloading verified runtime",
        );
    }
    file.flush().await.map_err(runtime_io)?;
    if count != asset.size {
        return Err(ErrorPayload::new(
            "runtime_size_mismatch",
            format!("Expected {} bytes, downloaded {count}.", asset.size),
            Some("Retry on a stable connection.".into()),
        ));
    }
    if hex::encode(digest.finalize()) != asset.sha256.to_ascii_lowercase() {
        return Err(ErrorPayload::new(
            "runtime_checksum_mismatch",
            "The runtime checksum did not match the pinned manifest.",
            Some("Delete the failed download and retry; do not run the archive.".into()),
        ));
    }
    Ok(path.to_path_buf())
}

fn extract_archive(archive_path: &Path, output: &Path) -> Result<(), ErrorPayload> {
    let file = File::open(archive_path).map_err(runtime_io)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| ErrorPayload::new("runtime_archive_invalid", error.to_string(), None))?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|error| {
            ErrorPayload::new("runtime_archive_invalid", error.to_string(), None)
        })?;
        let Some(relative) = entry.enclosed_name() else {
            return Err(ErrorPayload::new(
                "runtime_archive_unsafe",
                "The archive contains an unsafe path.",
                None,
            ));
        };
        let target = output.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&target).map_err(runtime_io)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(runtime_io)?;
        }
        let mut file = File::create(target).map_err(runtime_io)?;
        std::io::copy(&mut entry, &mut file).map_err(runtime_io)?;
    }
    Ok(())
}

fn download_error(error: reqwest::Error) -> ErrorPayload {
    ErrorPayload::new(
        "runtime_download_failed",
        error.to_string(),
        Some("Check the internet connection or use an offline verified archive.".into()),
    )
}
