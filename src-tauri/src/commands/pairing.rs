pub(crate) mod lifecycle;
pub(crate) mod reset;

pub use reset::reset_pairing;

use std::{net::SocketAddr, sync::Arc, time::Duration};

use tauri::{AppHandle, State};

use crate::{
    pairing::{now, PeerRecord},
    peer::{discover, local_ip_addresses, PeerClient},
    state::AppState,
    types::{ErrorPayload, NodeCapabilities},
};

pub(super) const PAIRING_PORT: u16 = 49_158;

#[tauri::command]
pub async fn connect_peer(
    manual_endpoint: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<NodeCapabilities, ErrorPayload> {
    let result = connect_peer_inner(manual_endpoint, &state).await;
    if result.is_ok() {
        lifecycle::start_persistent_peer_service(app).await;
    }
    result
}

async fn connect_peer_inner(
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
                endpoints.push(reject_local_endpoint(endpoint, &local_ip_addresses()?)?);
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
            "No SharedLocalLLM computer was discovered.",
            Some("Enter the other computer's Ethernet IPv4 address.".into()),
        ));
    }
    let mut last_error = None;
    for endpoint in endpoints {
        let client = PeerClient::new(endpoint, local.id.clone());
        let capabilities = serde_json::to_value(&local)
            .map_err(|error| ErrorPayload::new("capabilities_encode", error.to_string(), None))?;
        match client.connect(&local.name, capabilities).await {
            Ok(connected) => {
                let capabilities = connected.capabilities().await?;
                let mut node: NodeCapabilities =
                    serde_json::from_value(capabilities).map_err(|error| {
                        ErrorPayload::new(
                            "peer_capabilities_invalid",
                            error.to_string(),
                            Some("Ensure both computers run the same app version.".into()),
                        )
                    })?;
                if !connected.remote_device_id().is_empty() {
                    node.id = connected.remote_device_id().into();
                }
                if !connected.remote_device_name().is_empty() {
                    node.name = connected.remote_device_name().into();
                }
                node.online = true;
                let record = PeerRecord {
                    id: node.id.clone(),
                    name: node.name.clone(),
                    address: Some(endpoint.to_string()),
                    trusted_at: now(),
                    capabilities: Some(node.clone()),
                };
                {
                    let mut inner = state.lock()?;
                    inner.peers.clear();
                    inner.peers.push(record);
                }
                state.persist()?;
                if let Err(error) = state.refresh_models_shared() {
                    state.log("WARN", "model_fit_refresh_failed", &error.to_string());
                }
                state.peer.lock().await.client = Some(Arc::new(connected));
                state.log("INFO", "peer_connected", "Connected to the other computer");
                return Ok(node);
            }
            Err(error) => last_error = Some(error),
        }
    }
    let error = last_error.unwrap_or_else(|| {
        ErrorPayload::new(
            "peer_connect_failed",
            "The connection attempt failed.",
            None,
        )
    });
    state.log("WARN", "peer_connect_failed", &error.to_string());
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
            Some("Run ipconfig on the other computer and enter that computer's Ethernet IPv4 address.".into()),
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
        assert_eq!(
            parse_manual_endpoint(Some("10.10.10.2")).unwrap(),
            Some(SocketAddr::from((
                Ipv4Addr::new(10, 10, 10, 2),
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

    #[test]
    fn manual_endpoint_accepts_a_remote_address() {
        let endpoint = SocketAddr::from((Ipv4Addr::new(10, 10, 10, 2), PAIRING_PORT));
        let local_addresses = [IpAddr::V4(Ipv4Addr::new(10, 10, 10, 1))];
        assert_eq!(
            reject_local_endpoint(endpoint, &local_addresses).unwrap(),
            endpoint
        );
    }

    #[test]
    fn manual_endpoint_accepts_a_remote_link_local_address() {
        let endpoint = SocketAddr::from((Ipv4Addr::new(169, 254, 20, 8), PAIRING_PORT));
        let local_addresses = [IpAddr::V4(Ipv4Addr::new(169, 254, 179, 236))];
        assert_eq!(
            reject_local_endpoint(endpoint, &local_addresses).unwrap(),
            endpoint
        );
    }
}
