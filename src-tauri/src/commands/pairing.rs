mod network;
mod session;

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use tauri::{AppHandle, State};

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

pub(crate) use network::require_private_network;
use network::{close_firewall_lease, open_temporary_public_firewall_port, require_pairing_network};
use session::cleanup_pairing_session;

const PAIRING_PORT: u16 = 49_158;

#[tauri::command]
pub async fn generate_pairing_code(
    allow_public_network: bool,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<PairingCode, ErrorPayload> {
    let public_override = require_pairing_network(allow_public_network)?;
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
    let (previous_discovery, previous_server, previous_lease) = {
        let mut peer = state.peer.lock().await;
        peer.pairing_session_id = None;
        (
            peer.discovery.take(),
            peer.server.take(),
            peer.public_firewall_lease.take(),
        )
    };
    if let Some(previous) = previous_discovery {
        previous.shutdown().await;
    }
    if let Some(previous) = previous_server {
        previous.shutdown().await;
    }
    close_firewall_lease(previous_lease.as_deref());
    let server = PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, PAIRING_PORT)),
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
    let firewall_lease = if public_override {
        match open_temporary_public_firewall_port(server.address().port()).await {
            Ok(lease) => Some(lease),
            Err(error) => {
                server.shutdown().await;
                return Err(error);
            }
        }
    } else {
        None
    };
    let broadcaster = match DiscoveryBroadcaster::start(DiscoveryAnnouncement {
        protocol_version: 1,
        device_id: local.id,
        device_name: local.name,
        peer_port: server.address().port(),
    })
    .await
    {
        Ok(broadcaster) => broadcaster,
        Err(error) => {
            close_firewall_lease(firewall_lease.as_deref());
            server.shutdown().await;
            return Err(error);
        }
    };
    let mut pairing_completion = server.pairing_completion();
    let session_id = uuid::Uuid::new_v4().to_string();
    {
        let mut peer = state.peer.lock().await;
        peer.pairing_session_id = Some(session_id.clone());
        peer.server = Some(server);
        peer.discovery = Some(broadcaster);
        peer.public_firewall_lease = firewall_lease;
    }
    tokio::spawn(async move {
        let completed = tokio::select! {
            result = pairing_completion.changed() => result.is_ok() && *pairing_completion.borrow(),
            _ = tokio::time::sleep(Duration::from_secs(300)) => false,
        };
        cleanup_pairing_session(&app, &session_id, completed).await;
    });
    Ok(PairingCode {
        code: format!("{} {}", &code[..3], &code[3..]),
        expires_in_seconds: 300,
    })
}

#[tauri::command]
pub async fn pair_with_peer(
    code: String,
    allow_public_network: bool,
    manual_endpoint: Option<String>,
    state: State<'_, AppState>,
) -> Result<NodeCapabilities, ErrorPayload> {
    let _ = require_pairing_network(allow_public_network)?;
    let local = state.lock()?.local.clone();
    let mut endpoints = Vec::new();
    let manual_endpoint = parse_manual_endpoint(manual_endpoint.as_deref())?;
    if let Some(endpoint) = manual_endpoint {
        endpoints.push(endpoint);
    } else {
        if let Ok(endpoint) = std::env::var("SHARED_LOCAL_LLM_PEER_ENDPOINT") {
            if let Some(endpoint) = parse_manual_endpoint(Some(&endpoint))? {
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
    }
    endpoints.sort_unstable();
    endpoints.dedup();
    if endpoints.is_empty() {
        return Err(ErrorPayload::new(
            "peer_not_discovered",
            "No pairable SharedLocalLLM computer was discovered.",
            Some("Keep the pairing code visible on the other computer, or enter its Ethernet IPv4 address.".into()),
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

fn parse_manual_endpoint(value: Option<&str>) -> Result<Option<SocketAddr>, ErrorPayload> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if let Ok(endpoint) = value.parse::<SocketAddr>() {
        return Ok(Some(endpoint));
    }
    if let Ok(address) = value.parse::<std::net::IpAddr>() {
        return Ok(Some(SocketAddr::new(address, PAIRING_PORT)));
    }
    Err(ErrorPayload::new(
        "manual_peer_endpoint_invalid",
        format!("'{value}' is not a valid Ethernet IP address or IP:port."),
        Some(format!(
            "Enter the other computer's IPv4 address, for example 192.168.50.2. Port {PAIRING_PORT} is used automatically."
        )),
    ))
}

#[cfg(test)]
mod tests {
    use super::{parse_manual_endpoint, PAIRING_PORT};
    use std::net::{Ipv4Addr, SocketAddr};

    #[test]
    fn manual_ipv4_address_uses_the_standard_pairing_port() {
        assert_eq!(
            parse_manual_endpoint(Some("192.168.50.2")).unwrap(),
            Some(SocketAddr::from((Ipv4Addr::new(192, 168, 50, 2), PAIRING_PORT)))
        );
    }

    #[test]
    fn manual_endpoint_can_override_the_standard_port_and_reports_invalid_input() {
        assert_eq!(
            parse_manual_endpoint(Some("169.254.20.8:50123")).unwrap(),
            Some("169.254.20.8:50123".parse().unwrap())
        );
        let error = parse_manual_endpoint(Some("not-an-address")).unwrap_err();
        assert_eq!(error.code, "manual_peer_endpoint_invalid");
    }
}
