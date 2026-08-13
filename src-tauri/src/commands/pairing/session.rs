use tauri::{AppHandle, Manager};

use crate::state::AppState;

use super::network::close_firewall_lease;

pub(super) async fn cleanup_pairing_session(app: &AppHandle, session_id: &str, completed: bool) {
    let state = app.state::<AppState>();
    let cleanup = {
        let mut peer = state.peer.lock().await;
        if peer.pairing_session_id.as_deref() != Some(session_id) {
            return;
        }
        peer.pairing_session_id = None;
        (
            peer.discovery.take(),
            if completed { None } else { peer.server.take() },
            peer.public_firewall_lease.take(),
        )
    };
    if let Some(discovery) = cleanup.0 {
        discovery.shutdown().await;
    }
    if let Some(server) = cleanup.1 {
        server.shutdown().await;
    }
    close_firewall_lease(cleanup.2.as_deref());
}
