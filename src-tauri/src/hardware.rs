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
    #[serde(rename = "Virtual", default)]
    virtual_adapter: Option<bool>,
}

pub fn probe_local() -> NodeCapabilities {
    let mut system = System::new_all();
    system.refresh_all();
    let cpu = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(macos_cpu_fallback);
    let total = system.total_memory() as f64 / 1024_f64.powi(3);
    let available = system.available_memory() as f64 / 1024_f64.powi(3);
    let name = device_name();
    let gpu = probe_nvidia()
        .or_else(|| probe_apple_silicon(total, available))
        .unwrap_or_else(|| GpuInfo {
            name: "No GPU detected".into(),
            ..GpuInfo::default()
        });
    NodeCapabilities {
        id: "local-node".into(),
        name,
        online: true,
        role: "available".into(),
        cpu,
        ram_total_gb: total,
        ram_available_gb: available,
        gpu,
        adapter: probe_adapter().unwrap_or_else(|| {
            probe_macos_adapter().unwrap_or_else(|| NetworkAdapterInfo {
                name: "Unknown network adapter".into(),
                kind: "other".into(),
                link_speed_mbps: None,
            })
        }),
        cluster_status: None,
        cluster_model_id: None,
    }
}

fn device_name() -> String {
    if let Ok(name) = std::env::var("COMPUTERNAME") {
        if !name.trim().is_empty() {
            return name;
        }
    }
    if let Ok(name) = std::env::var("HOSTNAME") {
        if !name.trim().is_empty() {
            return name;
        }
    }
    // macOS: scutil ComputerName, then hostname.
    if let Some(name) =
        output_with_timeout("scutil", &["--get", "ComputerName"], Duration::from_secs(3))
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    {
        return name;
    }
    if let Some(name) = output_with_timeout("hostname", &[], Duration::from_secs(3))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return name;
    }
    "This computer".into()
}

fn macos_cpu_fallback() -> String {
    output_with_timeout(
        "sysctl",
        &["-n", "machdep.cpu.brand_string"],
        Duration::from_secs(3),
    )
    .map(|value| value.trim().to_owned())
    .filter(|value| !value.is_empty())
    .or_else(|| {
        output_with_timeout("sysctl", &["-n", "hw.model"], Duration::from_secs(3))
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
    .unwrap_or_else(|| "Unknown processor".into())
}

/// Apple Silicon (including M5) uses unified memory shared between CPU and GPU.
/// llama.cpp Metal backend can address that pool, so report it as GPU memory
/// instead of "No GPU detected". Conservative: 75% of currently available RAM.
fn probe_apple_silicon(ram_total_gb: f64, ram_available_gb: f64) -> Option<GpuInfo> {
    if std::env::consts::OS != "macos" {
        return None;
    }
    let arch = output_with_timeout("uname", &["-m"], Duration::from_secs(3))
        .map(|value| value.trim().to_owned())
        .unwrap_or_default();
    // Only claim unified GPU memory on Apple Silicon (arm64). Intel Macs without
    // NVIDIA keep the generic fallback so we never overstate VRAM.
    if arch != "arm64" {
        return None;
    }
    let chip = output_with_timeout("sysctl", &["-n", "hw.model"], Duration::from_secs(3))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "Apple Silicon".into());
    let os_version = output_with_timeout("sw_vers", &["-productVersion"], Duration::from_secs(3))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    // Sanity: unified pool cannot exceed physical RAM; clamp to measured values.
    let total = ram_total_gb.max(0.0);
    let available = (ram_available_gb.max(0.0) * 0.75).min(total);
    if total <= 0.0 {
        return None;
    }
    Some(GpuInfo {
        name: format!("{chip} GPU (unified memory, Metal)"),
        vram_total_gb: total,
        vram_available_gb: available,
        driver_version: os_version.map(|version| format!("macOS {version}")),
    })
}

fn probe_macos_adapter() -> Option<NetworkAdapterInfo> {
    if std::env::consts::OS != "macos" {
        return None;
    }
    // Default route interface, e.g. "en0".
    let interface = output_with_timeout("route", &["-n", "get", "default"], Duration::from_secs(3))
        .and_then(|output| {
            output.lines().find_map(|line| {
                let trimmed = line.trim();
                trimmed
                    .strip_prefix("interface:")
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty())
            })
        });
    let hardware_ports = output_with_timeout(
        "networksetup",
        &["-listallhardwareports"],
        Duration::from_secs(4),
    );
    if let (Some(interface), Some(ports)) = (interface, hardware_ports) {
        let mut current_port = String::new();
        for line in ports.lines() {
            let trimmed = line.trim();
            if let Some(port) = trimmed.strip_prefix("Hardware Port:") {
                current_port = port.trim().to_owned();
            } else if let Some(device) = trimmed.strip_prefix("Device:") {
                let device = device.trim();
                if device == interface {
                    let lowered = current_port.to_ascii_lowercase();
                    let kind = if lowered.contains("wi-fi") || lowered.contains("airport") {
                        "wifi"
                    } else if lowered.contains("ethernet") || lowered.contains("thunderbolt") {
                        "ethernet"
                    } else {
                        "other"
                    };
                    return Some(NetworkAdapterInfo {
                        name: format!("{current_port} ({device})"),
                        kind: kind.into(),
                        link_speed_mbps: None,
                    });
                }
            }
        }
        return Some(NetworkAdapterInfo {
            name: interface,
            kind: "other".into(),
            link_speed_mbps: None,
        });
    }
    None
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
    let script = "Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object Name,TransmitLinkSpeed,InterfaceDescription,Virtual | ConvertTo-Json -Compress";
    let stdout = output_with_timeout(
        "powershell.exe",
        &["-NoProfile", "-NonInteractive", "-Command", script],
        Duration::from_secs(5),
    )?;
    select_adapter(parse_adapters(&stdout)?).map(build_adapter_info)
}

fn parse_adapters(stdout: &str) -> Option<Vec<AdapterProbe>> {
    if let Ok(adapters) = serde_json::from_str::<Vec<AdapterProbe>>(stdout) {
        return Some(adapters);
    }
    serde_json::from_str::<AdapterProbe>(stdout)
        .ok()
        .map(|adapter| vec![adapter])
}

fn select_adapter(adapters: Vec<AdapterProbe>) -> Option<AdapterProbe> {
    adapters
        .into_iter()
        .filter(is_physical_adapter)
        .max_by_key(|adapter| adapter.transmit_link_speed.unwrap_or(0))
}

fn is_physical_adapter(adapter: &AdapterProbe) -> bool {
    if adapter.virtual_adapter == Some(true) {
        return false;
    }
    let text = format!(
        "{} {}",
        adapter.name.as_deref().unwrap_or_default(),
        adapter.interface_description.as_deref().unwrap_or_default()
    )
    .to_ascii_lowercase();
    ![
        "virtual",
        "hyper-v",
        "vethernet",
        "wsl",
        "loopback",
        "tap-windows",
        "wireguard",
        "tailscale",
        "zerotier",
        "bluetooth",
        "vpn",
        "pseudo-interface",
    ]
    .iter()
    .any(|keyword| text.contains(keyword))
}

fn build_adapter_info(adapter: AdapterProbe) -> NetworkAdapterInfo {
    let description = adapter.interface_description.unwrap_or_default();
    let lowered = format!(
        "{} {description}",
        adapter.name.as_deref().unwrap_or_default()
    )
    .to_ascii_lowercase();
    let kind = if lowered.contains("wi-fi") || lowered.contains("wireless") {
        "wifi"
    } else if lowered.contains("ethernet") {
        "ethernet"
    } else {
        "other"
    };
    NetworkAdapterInfo {
        name: adapter.name.unwrap_or(description),
        kind: kind.into(),
        link_speed_mbps: adapter
            .transmit_link_speed
            .map(|bits| bits as f64 / 1_000_000.0),
    }
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

#[cfg(test)]
mod tests {
    use super::{
        build_adapter_info, is_physical_adapter, parse_adapters, select_adapter, AdapterProbe,
    };

    fn probe(
        name: &str,
        description: &str,
        speed_mbps: u64,
        virtual_adapter: bool,
    ) -> AdapterProbe {
        AdapterProbe {
            name: Some(name.into()),
            transmit_link_speed: Some(speed_mbps * 1_000_000),
            interface_description: Some(description.into()),
            virtual_adapter: Some(virtual_adapter),
        }
    }

    #[test]
    fn parse_adapters_accepts_a_single_object_or_an_array() {
        let single = r#"{"Name":"Ethernet","TransmitLinkSpeed":1000000000,"InterfaceDescription":"Realtek","Virtual":false}"#;
        assert_eq!(parse_adapters(single).unwrap().len(), 1);

        let list = r#"[{"Name":"Ethernet","TransmitLinkSpeed":1000000000,"InterfaceDescription":"Realtek","Virtual":false},{"Name":"vEthernet (WSL)","TransmitLinkSpeed":10000000000,"InterfaceDescription":"Hyper-V Virtual Ethernet Adapter #2","Virtual":true}]"#;
        assert_eq!(parse_adapters(list).unwrap().len(), 2);
    }

    #[test]
    fn rejects_virtual_adapters_by_flag_and_name() {
        assert!(!is_physical_adapter(&probe(
            "Ethernet",
            "Hyper-V Virtual Ethernet Adapter",
            10000,
            false
        )));
        assert!(!is_physical_adapter(&probe(
            "vEthernet (WSL)",
            "Hyper-V Virtual Ethernet Adapter #2",
            10000,
            true
        )));
        assert!(!is_physical_adapter(&probe(
            "Ethernet 2",
            "Realtek Gaming 2.5GbE Family Controller",
            1000,
            true
        )));
        assert!(!is_physical_adapter(&probe(
            "Loopback",
            "Microsoft KM-TEST Loopback Adapter",
            1000,
            false
        )));
    }

    #[test]
    fn accepts_physical_adapters() {
        assert!(is_physical_adapter(&probe(
            "Ethernet",
            "Realtek Gaming 2.5GbE Family Controller",
            1000,
            false
        )));
        assert!(is_physical_adapter(&probe(
            "Wi-Fi",
            "RZ616 Wi-Fi 6E 160MHz",
            1200,
            false
        )));
    }

    #[test]
    fn selects_fastest_physical_adapter_and_skips_virtual_ones() {
        let adapters = vec![
            probe(
                "vEthernet (WSL)",
                "Hyper-V Virtual Ethernet Adapter #2",
                10000,
                true,
            ),
            probe(
                "Ethernet",
                "Realtek Gaming 2.5GbE Family Controller",
                1000,
                false,
            ),
            probe("Wi-Fi", "RZ616 Wi-Fi 6E 160MHz", 1200, false),
        ];
        let selected = select_adapter(adapters).unwrap();
        assert_eq!(selected.name.as_deref(), Some("Wi-Fi"));
    }

    #[test]
    fn maps_kind_and_link_speed() {
        let wired = build_adapter_info(probe(
            "Ethernet",
            "Realtek Gaming 2.5GbE Family Controller",
            1000,
            false,
        ));
        assert_eq!(wired.kind, "ethernet");
        assert_eq!(wired.link_speed_mbps, Some(1000.0));

        let wireless = build_adapter_info(probe("Wi-Fi", "RZ616 Wi-Fi 6E 160MHz", 1200, false));
        assert_eq!(wireless.kind, "wifi");
        assert_eq!(wireless.link_speed_mbps, Some(1200.0));
    }
}
