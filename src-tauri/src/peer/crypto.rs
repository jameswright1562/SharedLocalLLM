use sha2::{Digest, Sha256};
use snow::{params::NoiseParams, Builder, HandshakeState, TransportState};

use crate::types::ErrorPayload;

const PATTERN: &str = "Noise_NNpsk0_25519_ChaChaPoly_SHA256";

pub fn psk(code: &str) -> Result<[u8; 32], ErrorPayload> {
    let normalized: String = if code
        .chars()
        .all(|character| character.is_ascii_digit() || character.is_whitespace())
    {
        code.chars()
            .filter(|character| character.is_ascii_digit())
            .collect()
    } else {
        code.to_owned()
    };
    if normalized.is_empty() {
        return Err(ErrorPayload::new(
            "peer_credential_invalid",
            "The peer credential is empty.",
            None,
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"SharedLocalLLM peer credential v1\0");
    digest.update(normalized.as_bytes());
    Ok(digest.finalize().into())
}

pub fn initiator(code: &str) -> Result<HandshakeState, ErrorPayload> {
    build(code, true)
}

pub fn responder(code: &str) -> Result<HandshakeState, ErrorPayload> {
    build(code, false)
}

fn build(code: &str, initiator: bool) -> Result<HandshakeState, ErrorPayload> {
    let params: NoiseParams = PATTERN
        .parse()
        .map_err(|error| ErrorPayload::new("noise_pattern", format!("{error}"), None))?;
    let key = psk(code)?;
    let builder = Builder::new(params).psk(0, &key);
    let result = if initiator {
        builder.build_initiator()
    } else {
        builder.build_responder()
    };
    result.map_err(noise_error)
}

pub fn transport(handshake: HandshakeState) -> Result<TransportState, ErrorPayload> {
    handshake.into_transport_mode().map_err(noise_error)
}

pub fn noise_error(error: snow::Error) -> ErrorPayload {
    ErrorPayload::new(
        "peer_crypto",
        format!("Encrypted peer channel failed: {error}"),
        Some("Re-pair the computers (Forget the peer, then pair again) to refresh the shared channel key.".into()),
    )
}
