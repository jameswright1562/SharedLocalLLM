pub(crate) mod lifecycle;
mod network;
pub(crate) mod reset;
mod session;

pub use reset::reset_pairing;

use std::{net::SocketAddr, sync::Arc, time::Duration};

use tauri::{AppHandle, State};

use crate::{
    pairing::{now, PeerRecord},
    peer::{discover, local_ip_addresses, PeerClient},
    secrets,
    state::{peer_secret_path, AppState},
    types::{ErrorPayload, NodeCapabilities, PairingCode},
};

use lifecycle::start_peer_server;
pub(crate) use network::require_private_network;
use network::{close_firewall_lease, open_temporary_public_firewall_port, require_pairing_network};
use session::cleanup_pairing_session;

pub(super) const PAIRING_PORT: u16 = 49_158;

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
    let (server, broadcaster) = start_peer_server(&state, Some(code.clone())).await?;
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
            result = pairing_completion.changed() => {
                result.ok().and_then(|_| pairing_completion.borrow().clone())
            },
            _ = tokio::time::sleep(Duration::from_secs(300)) => None,
        };
        cleanup_pairing_session(&app, &session_id, completed).await;
    });
    state.log(
        "INFO",
        "pairing_code_ready",
        "Pairing is available for five minutes",
    );
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
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<NodeCapabilities, ErrorPayload> {
    let public_override = require_pairing_network(allow_public_network)?;
    let firewall_lease = if public_override {
        Some(open_temporary_public_firewall_port(PAIRING_PORT).await?)
    } else {
        None
    };
    let result = pair_with_peer_inner(code, manual_endpoint, &state).await;
    close_firewall_lease(firewall_lease.as_deref());
    if result.is_ok() {
        lifecycle::start_persistent_peer_service(app).await;
    }
    result
}

async fn pair_with_peer_inner(
    code: String,
    manual_endpoint: Option<String>,
    state: &AppState,
) -> Result<NodeCapabilities, ErrorPayload> {
    let local = state.lock()?.local.clone();
    let mut endpoints = Vec::new();
    let manual_endpoint = parse_manual_endpoint(manual_endpoint.as_deref())?;
    if let Some(endpoint) = manual_endpoint {
        endpoints.push(reject_local_endpoint(endpoint, &local_ip_addresses()?)?);
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
        let capabilities = serde_json::to_value(&local)
            .map_err(|error| ErrorPayload::new("capabilities_encode", error.to_string(), None))?;
        match PeerClient::pair(endpoint, &code, &local.id, &local.name, capabilities).await {
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
                let replaced_peer_ids = {
                    let mut inner = state.lock()?;
                    let ids = inner
                        .peers
                        .iter()
                        .filter(|peer| peer.id != record.id)
                        .map(|peer| peer.id.clone())
                        .collect::<Vec<_>>();
                    inner.peers.clear();
                    inner.peers.push(record);
                    ids
                };
                for replaced_peer_id in replaced_peer_ids {
                    secrets::remove(&peer_secret_path(&replaced_peer_id))?;
                }
                state.persist()?;
                if let Err(error) = state.refresh_models_shared() {
                    state.log("WARN", "model_fit_refresh_failed", &error.to_string());
                }
                state.peer.lock().await.client = Some(Arc::new(client));
                state.log(
                    "INFO",
                    "pairing_complete",
                    "Saved the trusted peer and channel key",
                );
                return Ok(node);
            }
            Err(error) => last_error = Some(error),
        }
    }
    let error = last_error.unwrap_or_else(|| {
        ErrorPayload::new("pairing_failed", "The pairing attempt failed.", None)
    });
    state.log("WARN", "pairing_failed", &error.to_string());
    Err(error)
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

fn reject_local_endpoint(
    endpoint: SocketAddr,
    local_addresses: &[std::net::IpAddr],
) -> Result<SocketAddr, ErrorPayload> {
    if local_addresses.contains(&endpoint.ip()) {
        return Err(ErrorPayload::new(
            "manual_peer_endpoint_is_local",
            format!("{} belongs to this computer, not the other computer.", endpoint.ip()),
            Some("Run ipconfig on the computer displaying the code and enter that computer's Ethernet IPv4 address.".into()),
        ));
    }
    Ok(endpoint)
}

#[cfg(test)]
mod tests {
    use super::{parse_manual_endpoint, reject_local_endpoint, PAIRING_PORT};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    #[test]
    fn manual_ipv4_address_uses_the_standard_pairing_port() {
        assert_eq!(
            parse_manual_endpoint(Some("192.168.50.2")).unwrap(),
            Some(SocketAddr::from((
                Ipv4Addr::new(192, 168, 50, 2),
                PAIRING_PORT
            )))
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

    #[test]
    fn manual_endpoint_rejects_an_address_owned_by_this_computer() {
        let endpoint = SocketAddr::from((Ipv4Addr::new(169, 254, 179, 236), PAIRING_PORT));
        let local_addresses = [IpAddr::V4(Ipv4Addr::new(169, 254, 179, 236))];
        let error = reject_local_endpoint(endpoint, &local_addresses).unwrap_err();
        assert_eq!(error.code, "manual_peer_endpoint_is_local");
    }
}
