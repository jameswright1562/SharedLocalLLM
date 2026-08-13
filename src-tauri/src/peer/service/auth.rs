use std::time::Instant;

use super::connection::ServerState;
use crate::{peer::protocol::ClientHello, types::ErrorPayload};

pub(super) async fn credential_for(
    state: &ServerState,
    hello: &ClientHello,
) -> Result<String, ErrorPayload> {
    if hello.pairing {
        if Instant::now() > state.pairing_expires_at {
            return Err(ErrorPayload::new(
                "pairing_code_expired",
                "The pairing code expired.",
                Some("Generate a new code.".into()),
            ));
        }
        state.pairing_code.lock().await.clone().ok_or_else(|| {
            ErrorPayload::new(
                "pairing_unavailable",
                "This peer is not currently showing a pairing code.",
                None,
            )
        })
    } else {
        state
            .trusted
            .lock()
            .await
            .get(&hello.device_id)
            .cloned()
            .ok_or_else(untrusted_error)
    }
}

pub(super) async fn ensure_trusted(
    state: &ServerState,
    device_id: &str,
) -> Result<(), ErrorPayload> {
    if state.trusted.lock().await.contains_key(device_id) {
        Ok(())
    } else {
        Err(untrusted_error())
    }
}

fn untrusted_error() -> ErrorPayload {
    ErrorPayload::new(
        "peer_untrusted",
        "The peer identity is not trusted.",
        Some("Pair the computers again.".into()),
    )
}
