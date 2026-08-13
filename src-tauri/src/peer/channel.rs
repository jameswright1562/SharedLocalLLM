use serde::{Deserialize, Serialize};

use super::{crypto, protocol};
use crate::types::ErrorPayload;

pub(crate) async fn send_encrypted<W, T>(
    writer: &mut W,
    noise: &mut snow::TransportState,
    value: &T,
) -> Result<(), ErrorPayload>
where
    W: tokio::io::AsyncWrite + Unpin,
    T: Serialize,
{
    let plain = serde_json::to_vec(value)
        .map_err(|error| ErrorPayload::new("peer_protocol", error.to_string(), None))?;
    let mut encrypted = vec![0; plain.len() + 16];
    let count = noise
        .write_message(&plain, &mut encrypted)
        .map_err(crypto::noise_error)?;
    encrypted.truncate(count);
    protocol::write_plain(writer, &encrypted).await
}

pub(crate) async fn receive_encrypted<R, T>(
    reader: &mut R,
    noise: &mut snow::TransportState,
) -> Result<T, ErrorPayload>
where
    R: tokio::io::AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let encrypted: Vec<u8> = protocol::read_plain(reader).await?;
    let mut plain = vec![0; encrypted.len()];
    let count = noise
        .read_message(&encrypted, &mut plain)
        .map_err(crypto::noise_error)?;
    serde_json::from_slice(&plain[..count])
        .map_err(|error| ErrorPayload::new("peer_protocol", error.to_string(), None))
}

pub(crate) async fn send_encrypted_bytes<W>(
    writer: &mut W,
    noise: &mut snow::TransportState,
    plain: &[u8],
) -> Result<(), ErrorPayload>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut encrypted = vec![0; plain.len() + 16];
    let count = noise
        .write_message(plain, &mut encrypted)
        .map_err(crypto::noise_error)?;
    encrypted.truncate(count);
    protocol::write_plain(writer, &encrypted).await
}

pub(crate) async fn receive_encrypted_bytes<R>(
    reader: &mut R,
    noise: &mut snow::TransportState,
) -> Result<Vec<u8>, ErrorPayload>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let encrypted: Vec<u8> = protocol::read_plain(reader).await?;
    let mut plain = vec![0; encrypted.len()];
    let count = noise
        .read_message(&encrypted, &mut plain)
        .map_err(crypto::noise_error)?;
    plain.truncate(count);
    Ok(plain)
}
