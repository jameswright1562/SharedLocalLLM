use tauri::{AppHandle, Emitter, Manager};

use crate::{peer::PeerPairingEvent, state::AppState};

use super::lifecycle;

pub(super) async fn cleanup_pairing_session(
    app: &AppHandle,
    session_id: &str,
    completed: Option<PeerPairingEvent>,
) {
    let state = app.state::<AppState>();
    let paired = completed.is_some();
    if let Some(event) = completed {
        if let Err(error) = lifecycle::persist_incoming_pair(&state, event) {
            state.log("ERROR", "pairing_persist_failed", &error.to_string());
        } else {
            let _ = app.emit("pairing-complete", true);
        }
    }
    let cleanup = {
        let mut peer = state.peer.lock().await;
        if peer.pairing_session_id.as_deref() != Some(session_id) {
            return;
        }
        peer.pairing_session_id = None;
        (peer.discovery.take(), peer.server.take())
    };
    if let Some(discovery) = cleanup.0 {
        discovery.shutdown().await;
    }
    if let Some(server) = cleanup.1 {
        server.shutdown().await;
    }
    if !paired {
        state.log(
            "INFO",
            "pairing_expired",
            "Pairing closed after five minutes",
        );
    }
    lifecycle::start_persistent_peer_service(app.clone()).await;
}
