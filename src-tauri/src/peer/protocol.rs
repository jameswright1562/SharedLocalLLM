use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::types::ErrorPayload;

pub const PROTOCOL_VERSION: u16 = 3;
pub const MAX_FRAME: usize = 1024 * 1024;

pub fn check_version(version: u16) -> Result<(), ErrorPayload> {
    if version == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(ErrorPayload::new(
            "peer_version",
            "The peer protocol versions do not match.",
            Some("Upgrade SharedLocalLLM on both computers.".into()),
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Pair {
        version: u16,
        device_id: String,
        device_name: String,
        capabilities: Value,
    },
    Heartbeat {
        version: u16,
        device_id: String,
    },
    Capabilities,
    Benchmark {
        size: u32,
    },
    RpcTunnel,
    StopWorker,
    Models,
    ProxyChat {
        messages: serde_json::Value,
        settings: serde_json::Value,
        images: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientHello {
    pub device_id: String,
    pub pairing: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Paired {
        device_id: String,
        device_name: String,
        channel_key: String,
    },
    Heartbeat,
    Capabilities {
        value: Value,
    },
    Benchmark {
        size: u32,
    },
    RpcReady,
    WorkerStopped,
    Models {
        models: Value,
    },
    ProxyChat {
        content: String,
    },
    Error {
        code: String,
        message: String,
    },
}

pub async fn write_plain<W: AsyncWrite + Unpin, T: Serialize>(
    writer: &mut W,
    value: &T,
) -> Result<(), ErrorPayload> {
    let payload = serde_json::to_vec(value).map_err(protocol_error)?;
    if payload.len() > MAX_FRAME {
        return Err(ErrorPayload::new(
            "peer_frame_large",
            "Peer frame exceeds the one MiB safety limit.",
            None,
        ));
    }
    writer
        .write_u32(payload.len() as u32)
        .await
        .map_err(io_error)?;
    writer.write_all(&payload).await.map_err(io_error)?;
    writer.flush().await.map_err(io_error)
}

pub async fn read_plain<R: AsyncRead + Unpin, T: for<'de> Deserialize<'de>>(
    reader: &mut R,
) -> Result<T, ErrorPayload> {
    let size = reader.read_u32().await.map_err(io_error)? as usize;
    if size > MAX_FRAME {
        return Err(ErrorPayload::new(
            "peer_frame_large",
            "Peer frame exceeds the one MiB safety limit.",
            None,
        ));
    }
    let mut payload = vec![0; size];
    reader.read_exact(&mut payload).await.map_err(io_error)?;
    serde_json::from_slice(&payload).map_err(protocol_error)
}

fn protocol_error(error: serde_json::Error) -> ErrorPayload {
    ErrorPayload::new("peer_protocol", error.to_string(), None)
}
pub fn io_error(error: std::io::Error) -> ErrorPayload {
    ErrorPayload::new(
        "peer_io",
        error.to_string(),
        Some("Confirm both computers remain connected and reachable on the same network.".into()),
    )
}
