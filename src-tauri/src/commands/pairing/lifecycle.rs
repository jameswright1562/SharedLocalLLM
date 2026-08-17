use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use tauri::{AppHandle, Manager};

use crate::{
    pairing::{now, PeerRecord},
    peer::{
        discover, DiscoveryAnnouncement, DiscoveryBroadcaster, PeerClient, PeerPairingEvent,
        PeerServer, PeerServerConfig, TrustedPeer, DISCOVERY_PORT,
    },
    runtime, secrets,
    state::{peer_secret_path, AppState},
    types::{ErrorPayload, NodeCapabilities},
};

use super::PAIRING_PORT;

pub(super) async fn start_peer_server(
    state: &AppState,
    pairing_code: Option<String>,
) -> Result<(PeerServer, DiscoveryBroadcaster), ErrorPayload> {
    let (local, peers, api_key, api_port, catalogue) = {
        let inner = state.lock()?;
        let mut local = inner.local.clone();
        local.cluster_status = Some(inner.cluster.status.clone());
        local.cluster_model_id = inner.cluster.model_id.clone();
        (
            local,
            inner.peers.clone(),
            inner.api_key.clone(),
            inner.api_port,
            serde_json::to_value(&inner.models).unwrap_or_default(),
        )
    };
    let capabilities = serde_json::to_value(&local)
        .map_err(|error| ErrorPayload::new("capabilities_encode", error.to_string(), None))?;
    let mut trusted_peers = Vec::new();
    for peer in peers {
        match secrets::load(&peer_secret_path(&peer.id))? {
            Some(bytes) => match String::from_utf8(bytes) {
                Ok(channel_key) => trusted_peers.push(TrustedPeer {
                    device_id: peer.id,
                    device_name: peer.name,
                    channel_key,
                }),
                Err(_) => state.log(
                    "ERROR",
                    "peer_secret_invalid",
                    "A protected peer credential is not valid UTF-8",
                ),
            },
            None => state.log(
                "WARN",
                "peer_secret_missing",
                "A saved peer cannot reconnect until it is paired again",
            ),
        }
    }
    let rpc_path = runtime::runtime_root()
        .join("current")
        .join("ggml-rpc-server.exe");
    let server = PeerServer::start(PeerServerConfig {
        bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, PAIRING_PORT)),
        device_id: local.id.clone(),
        device_name: local.name.clone(),
        capabilities,
        pairing_code,
        trusted_peers,
        rpc_binary: rpc_path.is_file().then_some(rpc_path),
        rpc_override: None,
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
    match start_peer_server(&state, None).await {
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

pub(super) fn persist_incoming_pair(
    state: &AppState,
    event: PeerPairingEvent,
) -> Result<(), ErrorPayload> {
    let mut capabilities: NodeCapabilities = serde_json::from_value(event.capabilities)
        .map_err(|error| ErrorPayload::new("peer_capabilities_invalid", error.to_string(), None))?;
    capabilities.id = event.device_id.clone();
    capabilities.name = event.device_name.clone();
    capabilities.online = true;
    capabilities.role = "worker".into();
    secrets::store(
        &peer_secret_path(&event.device_id),
        event.channel_key.as_bytes(),
    )?;
    let endpoint = SocketAddr::new(event.source.ip(), PAIRING_PORT);
    let record = PeerRecord {
        id: event.device_id,
        name: event.device_name,
        address: Some(endpoint.to_string()),
        trusted_at: now(),
        capabilities: Some(capabilities),
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
    state.log(
        "INFO",
        "pairing_accepted",
        "Saved the trusted peer on the code host",
    );
    Ok(())
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
                    "The trusted peer answered a heartbeat",
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
    let mut client = state.peer_client().await?;
    if let Err(first_error) = client.heartbeat().await {
        state.peer.lock().await.client = None;
        let Some((_, endpoint)) = discover(Duration::from_millis(900))
            .await?
            .into_iter()
            .find(|(announcement, _)| announcement.device_id == peer.id)
        else {
            return Err(first_error);
        };
        let discovered = Arc::new(PeerClient::trusted(
            endpoint,
            client.channel_key().to_owned(),
            state.lock()?.local.id.clone(),
        ));
        discovered.heartbeat().await?;
        if let Some(saved) = state.lock()?.peers.first_mut() {
            saved.address = Some(endpoint.to_string());
        }
        state.persist()?;
        client = discovered;
    }
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
