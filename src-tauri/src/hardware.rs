use std::process::Command;

use serde::Deserialize;
use sysinfo::System;

use crate::types::{GpuInfo, NetworkAdapterInfo, NodeCapabilities};

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct AdapterProbe {
    name: Option<String>,
    link_speed: Option<u64>,
    interface_description: Option<String>,
}

pub fn probe_local() -> NodeCapabilities {
    let mut system = System::new_all();
    system.refresh_all();
    let cpu = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown processor".into());
    let total = system.total_memory() as f64 / 1024_f64.powi(3);
    let available = system.available_memory() as f64 / 1024_f64.powi(3);
    let name = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "This computer".into());
    NodeCapabilities {
        id: "local-node".into(),
        name,
        online: true,
        role: "available".into(),
        cpu,
        ram_total_gb: total,
        ram_available_gb: available,
        gpu: probe_nvidia().unwrap_or_else(|| GpuInfo {
            name: "No NVIDIA CUDA GPU detected".into(),
            ..GpuInfo::default()
        }),
        adapter: probe_adapter().unwrap_or_else(|| NetworkAdapterInfo {
            name: "Unknown network adapter".into(),
            kind: "other".into(),
            link_speed_mbps: None,
        }),
    }
}

fn probe_nvidia() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .to_owned();
    let parts: Vec<_> = line.split(',').map(str::trim).collect();
    if parts.len() < 4 {
        return None;
    }
    Some(GpuInfo {
        name: parts[0].into(),
        vram_total_gb: parts[1].parse::<f64>().ok()? / 1024.0,
        vram_available_gb: parts[2].parse::<f64>().ok()? / 1024.0,
        driver_version: Some(parts[3].into()),
    })
}

fn probe_adapter() -> Option<NetworkAdapterInfo> {
    let script = "Get-NetAdapter | Where-Object Status -eq 'Up' | Sort-Object LinkSpeed -Descending | Select-Object -First 1 Name,LinkSpeed,InterfaceDescription | ConvertTo-Json -Compress";
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let probe: AdapterProbe = serde_json::from_slice(&output.stdout).ok()?;
    let description = probe.interface_description.unwrap_or_default();
    let lowered = format!(
        "{} {description}",
        probe.name.as_deref().unwrap_or_default()
    )
    .to_ascii_lowercase();
    let kind = if lowered.contains("wi-fi") || lowered.contains("wireless") {
        "wifi"
    } else if lowered.contains("ethernet") {
        "ethernet"
    } else {
        "other"
    };
    Some(NetworkAdapterInfo {
        name: probe.name.unwrap_or(description),
        kind: kind.into(),
        link_speed_mbps: probe.link_speed.map(|bits| bits as f64 / 1_000_000.0),
    })
}
