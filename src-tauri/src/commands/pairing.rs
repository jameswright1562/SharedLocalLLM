use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use tauri::State;

use crate::{
    pairing::{now, PeerRecord},
    peer::{
        discover, DiscoveryAnnouncement, DiscoveryBroadcaster, PeerClient, PeerServer,
        PeerServerConfig,
    },
    runtime, secrets,
    state::{peer_secret_path, AppState},
    types::{ErrorPayload, NodeCapabilities, PairingCode},
};

#[tauri::command]
pub async fn generate_pairing_code(
    state: State<'_, AppState>,
) -> Result<PairingCode, ErrorPayload> {
    require_private_network()?;
    let code = state
        .pairing
        .lock()
        .map_err(|_| ErrorPayload::new("pairing_state", "Pairing state is unavailable.", None))?
        .generate();
    let local = state.lock()?.local.clone();
    let rpc_path = runtime::runtime_root()
        .join("current")
        .join("ggml-rpc-server.exe");
    let rpc_command = rpc_path.is_file().then(|| {
        vec![
            rpc_path.to_string_lossy().into_owned(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            "50052".into(),
        ]
    });
    let server = PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
        device_id: local.id.clone(),
        device_name: local.name.clone(),
        capabilities: serde_json::to_value(&local)
            .map_err(|error| ErrorPayload::new("capabilities_encode", error.to_string(), None))?,
        pairing_code: Some(code.clone()),
        trusted_peers: vec![],
        rpc_target: SocketAddr::from((Ipv4Addr::LOCALHOST, 50052)),
        rpc_command,
    })
    .await?;
    let broadcaster = DiscoveryBroadcaster::start(DiscoveryAnnouncement {
        protocol_version: 1,
        device_id: local.id,
        device_name: local.name,
        peer_port: server.address().port(),
    })
    .await?;
    let mut peer = state.peer.lock().await;
    if let Some(previous) = peer.discovery.take() {
        previous.shutdown().await;
    }
    if let Some(previous) = peer.server.take() {
        previous.shutdown().await;
    }
    peer.server = Some(server);
    peer.discovery = Some(broadcaster);
    Ok(PairingCode {
        code: format!("{} {}", &code[..3], &code[3..]),
        expires_in_seconds: 300,
    })
}

#[tauri::command]
pub async fn pair_with_peer(
    code: String,
    state: State<'_, AppState>,
) -> Result<NodeCapabilities, ErrorPayload> {
    require_private_network()?;
    let local = state.lock()?.local.clone();
    let mut endpoints = Vec::new();
    if let Ok(endpoint) = std::env::var("SHARED_LOCAL_LLM_PEER_ENDPOINT") {
        if let Ok(endpoint) = endpoint.parse() {
            endpoints.push(endpoint);
        }
    }
    endpoints.extend(
        discover(Duration::from_secs(5))
            .await?
            .into_iter()
            .filter(|(announcement, _)| announcement.device_id != local.id)
            .map(|(_, endpoint)| endpoint),
    );
    endpoints.sort_unstable();
    endpoints.dedup();
    if endpoints.is_empty() {
        return Err(ErrorPayload::new(
            "peer_not_discovered",
            "No pairable SharedLocalLLM computer was discovered.",
            Some("Keep the pairing code visible on the other computer or set a manual peer endpoint.".into()),
        ));
    }
    let mut last_error = None;
    for endpoint in endpoints {
        match PeerClient::pair(endpoint, &code, &local.id, &local.name).await {
            Ok(client) => {
                let capabilities = client.capabilities().await?;
                let mut node: NodeCapabilities =
                    serde_json::from_value(capabilities).map_err(|error| {
                        ErrorPayload::new(
                            "peer_capabilities_invalid",
                            error.to_string(),
                            Some("Ensure both computers run the same app version.".into()),
                        )
                    })?;
                if !client.remote_device_id().is_empty() {
                    node.id = client.remote_device_id().into();
                }
                if !client.remote_device_name().is_empty() {
                    node.name = client.remote_device_name().into();
                }
                node.online = true;
                secrets::store(&peer_secret_path(&node.id), client.channel_key().as_bytes())?;
                let record = PeerRecord {
                    id: node.id.clone(),
                    name: node.name.clone(),
                    address: Some(endpoint.to_string()),
                    trusted_at: now(),
                    capabilities: Some(node.clone()),
                };
                {
                    let mut inner = state.lock()?;
                    inner.peers.retain(|peer| peer.id != record.id);
                    inner.peers.push(record);
                }
                state.persist()?;
                state.peer.lock().await.client = Some(Arc::new(client));
                return Ok(node);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        ErrorPayload::new("pairing_failed", "The pairing attempt failed.", None)
    }))
}

fn require_private_network() -> Result<(), ErrorPayload> {
    #[cfg(windows)]
    {
        let output = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", "@(Get-NetConnectionProfile | Where-Object IPv4Connectivity -ne 'Disconnected' | Select-Object -ExpandProperty NetworkCategory) -join ','"])
            .output()
            .map_err(|error| ErrorPayload::new("network_profile_probe", error.to_string(), None))?;
        let profiles = String::from_utf8_lossy(&output.stdout);
        if !profiles.split(',').any(|profile| {
            profile.trim().eq_ignore_ascii_case("Private")
                || profile.trim().eq_ignore_ascii_case("DomainAuthenticated")
        }) {
            return Err(ErrorPayload::new(
                "private_network_required",
                "Pairing is available only on a Windows Private network profile.",
                Some("Change the trusted LAN profile to Private in Windows Settings.".into()),
            ));
        }
    }
    Ok(())
}
