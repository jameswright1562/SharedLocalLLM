use std::{
    io::Read,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use serde::Deserialize;
use sysinfo::System;

use crate::types::{GpuInfo, NetworkAdapterInfo, NodeCapabilities};

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct AdapterProbe {
    name: Option<String>,
    transmit_link_speed: Option<u64>,
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
        cluster_status: None,
        cluster_model_id: None,
    }
}

fn probe_nvidia() -> Option<GpuInfo> {
    let stdout = output_with_timeout(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        Duration::from_secs(4),
    )?;
    let mut names = Vec::new();
    let mut total = 0.0;
    let mut available = 0.0;
    let mut driver = None;
    for line in stdout.lines() {
        let parts: Vec<_> = line.split(',').map(str::trim).collect();
        if parts.len() < 4 {
            continue;
        }
        names.push(parts[0].to_owned());
        total += parts[1].parse::<f64>().ok()? / 1024.0;
        available += parts[2].parse::<f64>().ok()? / 1024.0;
        driver = Some(parts[3].to_owned());
    }
    if names.is_empty() {
        return None;
    }
    Some(GpuInfo {
        name: names.join(" + "),
        vram_total_gb: total,
        vram_available_gb: available,
        driver_version: driver,
    })
}

fn probe_adapter() -> Option<NetworkAdapterInfo> {
    let script = "Get-NetAdapter | Where-Object Status -eq 'Up' | Sort-Object TransmitLinkSpeed -Descending | Select-Object -First 1 Name,TransmitLinkSpeed,InterfaceDescription | ConvertTo-Json -Compress";
    let stdout = output_with_timeout(
        "powershell.exe",
        &["-NoProfile", "-NonInteractive", "-Command", script],
        Duration::from_secs(5),
    )?;
    let probe: AdapterProbe = serde_json::from_str(&stdout).ok()?;
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
        link_speed_mbps: probe
            .transmit_link_speed
            .map(|bits| bits as f64 / 1_000_000.0),
    })
}

pub(crate) fn output_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                let mut buf = String::new();
                child.stdout.take()?.read_to_string(&mut buf).ok()?;
                return Some(buf);
            }
            Ok(Some(_)) => return None,
            Ok(None) if started.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(40)),
            Err(_) => return None,
        }
    }
}
