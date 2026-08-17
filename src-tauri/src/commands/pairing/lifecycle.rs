use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use tauri::{AppHandle, Manager};

use crate::{
    pairing::{now, PeerRecord},
    peer::{
        DiscoveryAnnouncement, DiscoveryBroadcaster, PeerConnectedEvent, PeerServer,
        PeerServerConfig, DISCOVERY_PORT,
    },
    state::AppState,
    types::{ErrorPayload, NodeCapabilities},
};

use super::PAIRING_PORT;

pub(super) async fn start_peer_server(
    state: &AppState,
) -> Result<(PeerServer, DiscoveryBroadcaster), ErrorPayload> {
    let (local, api_key, api_port, catalogue) = {
        let inner = state.lock()?;
        let mut local = inner.local.clone();
        local.cluster_status = Some(inner.cluster.status.clone());
        local.cluster_model_id = inner.cluster.model_id.clone();
        (
            local,
            inner.api_key.clone(),
            inner.api_port,
            serde_json::to_value(&inner.models).unwrap_or_default(),
        )
    };
    let capabilities = serde_json::to_value(&local)
        .map_err(|error| ErrorPayload::new("capabilities_encode", error.to_string(), None))?;
    let server = PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, PAIRING_PORT)),

        device_id: local.id.clone(),
        device_name: local.name.clone(),

        capabilities,

        rpc_target: SocketAddr::from((Ipv4Addr::LOCALHOST, 50052)),

        catalogue,
        api_key,
        api_port,
    })
    .await?;
    let broadcaster = DiscoveryBroadcaster::start(DiscoveryAnnouncement {
        protocol_version: 3,
        device_id: local.id,
        device_name: local.name,
        peer_port: server.address().port(),
    })
    .await?;
    Ok((server, broadcaster))
}

pub async fn start_persistent_peer_service(app: AppHandle) {
    let state = app.state::<AppState>();
    if let Ok(executable) = std::env::current_exe() {
        if let Err(error) =
            crate::firewall::ensure_peer_firewall_rules(&executable, PAIRING_PORT, DISCOVERY_PORT)
                .await
        {
            state.log("WARN", "firewall_rule_failed", &error);
        }
    }
    let previous = {
        let mut peer = state.peer.lock().await;
        (peer.discovery.take(), peer.server.take())
    };
    if let Some(discovery) = previous.0 {
        discovery.shutdown().await;
    }
    if let Some(server) = previous.1 {
        server.shutdown().await;
    }
    match start_peer_server(&state).await {
        Ok((server, broadcaster)) => {
            let address = server.address().to_string();
            let mut peer = state.peer.lock().await;
            peer.server = Some(server);
            peer.discovery = Some(broadcaster);
            drop(peer);
            state.log(
                "INFO",
                "peer_listener_ready",
                &format!("Listening on {address}"),
            );
        }
        Err(error) => state.log("ERROR", "peer_listener_failed", &error.to_string()),
    }
}

pub async fn drain_peer_connects(state: &AppState) {
    let events = {
        let peer = state.peer.lock().await;
        match peer.server.as_ref() {
            Some(server) => server.take_connected(),
            None => Vec::new(),
        }
    };
    for event in events {
        persist_connected_peer(state, event);
    }
}

fn persist_connected_peer(state: &AppState, event: PeerConnectedEvent) {
    let mut capabilities: NodeCapabilities = match serde_json::from_value(event.capabilities) {
        Ok(capabilities) => capabilities,
        Err(error) => {
            state.log("WARN", "peer_capabilities_invalid", &error.to_string());
            return;
        }
    };
    capabilities.id = event.device_id.clone();
    capabilities.name = event.device_name.clone();
    capabilities.online = true;
    capabilities.role = "worker".into();
    let endpoint = SocketAddr::new(event.source.ip(), PAIRING_PORT);
    let record = PeerRecord {
        id: event.device_id,
        name: event.device_name,
        address: Some(endpoint.to_string()),
        trusted_at: now(),
        capabilities: Some(capabilities),
    };
    if let Ok(mut inner) = state.lock() {
        inner.peers.clear();
        inner.peers.push(record);
    }
    if let Err(error) = state.persist() {
        state.log("WARN", "peer_persist_failed", &error.to_string());
    }
    if let Err(error) = state.refresh_models_shared() {
        state.log("WARN", "model_fit_refresh_failed", &error.to_string());
    }
    state.log("INFO", "peer_connected", "The other computer connected");
}

pub async fn refresh_peer_status(state: &AppState) {
    let peer = match state.lock() {
        Ok(inner) => inner.peers.first().cloned(),
        Err(_) => return,
    };
    let Some(peer) = peer else { return };
    let was_online = peer
        .capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.online);
    let result = connect_and_refresh(state, &peer).await;
    match result {
        Ok(capabilities) => {
            if let Ok(mut inner) = state.lock() {
                if let Some(saved) = inner.peers.first_mut() {
                    saved.name = capabilities.name.clone();
                    saved.capabilities = Some(capabilities);
                }
            }
            if !was_online {
                state.log(
                    "INFO",
                    "peer_reconnected",
                    "The other computer answered a heartbeat",
                );
            }
        }
        Err(error) => {
            state.peer.lock().await.client = None;
            if let Ok(mut inner) = state.lock() {
                if let Some(capabilities) = inner
                    .peers
                    .first_mut()
                    .and_then(|saved| saved.capabilities.as_mut())
                {
                    capabilities.online = false;
                }
            }
            if was_online {
                state.log("WARN", "peer_disconnected", &error.to_string());
            }
        }
    }
}

async fn connect_and_refresh(
    state: &AppState,
    peer: &PeerRecord,
) -> Result<NodeCapabilities, ErrorPayload> {
    let client = state.peer_client().await?;
    client.heartbeat().await?;
    let mut capabilities: NodeCapabilities = serde_json::from_value(client.capabilities().await?)
        .map_err(|error| {
        ErrorPayload::new("peer_capabilities_invalid", error.to_string(), None)
    })?;
    capabilities.id = peer.id.clone();
    capabilities.name = peer.name.clone();
    capabilities.online = true;
    capabilities.role = "worker".into();
    state.peer.lock().await.client = Some(Arc::clone(&client));
    Ok(capabilities)
}
