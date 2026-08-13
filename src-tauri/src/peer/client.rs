use std::{
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpStream;

use super::{
    channel::{receive_encrypted, send_encrypted},
    crypto,
    protocol::{self, ClientHello, Request, Response, PROTOCOL_VERSION},
    tunnel::RpcForwarder,
};
use crate::types::ErrorPayload;

#[derive(Clone)]
pub struct PeerClient {
    endpoint: SocketAddr,
    device_id: String,
    credential: String,
    remote_device_id: String,
    remote_device_name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub throughput_mbps: f64,
    pub latency_median_ms: f64,
    pub latency_p95_ms: f64,
    pub samples: usize,
}

impl PeerClient {
    pub async fn pair(
        endpoint: SocketAddr,
        code: &str,
        device_id: &str,
        device_name: &str,
        capabilities: Value,
    ) -> Result<Self, ErrorPayload> {
        let normalized: String = code
            .chars()
            .filter(|character| character.is_ascii_digit())
            .collect();
        if normalized.len() != 6 {
            return Err(ErrorPayload::new(
                "pairing_code_invalid",
                "Pairing codes contain exactly six digits.",
                None,
            ));
        }
        let mut client = Self {
            endpoint,
            device_id: device_id.into(),
            credential: normalized,
            remote_device_id: String::new(),
            remote_device_name: String::new(),
        };
        match client
            .request(Request::Pair {
                version: PROTOCOL_VERSION,
                device_id: device_id.into(),
                device_name: device_name.into(),
                capabilities,
            })
            .await?
        {
            Response::Paired {
                device_id,
                device_name,
                channel_key,
            } => {
                client.credential = channel_key;
                client.remote_device_id = device_id;
                client.remote_device_name = device_name;
                Ok(client)
            }
            response => Err(response_error(response)),
        }
    }
    pub fn trusted(endpoint: SocketAddr, channel_key: String, device_id: String) -> Self {
        Self {
            endpoint,
            device_id,
            credential: channel_key,
            remote_device_id: String::new(),
            remote_device_name: String::new(),
        }
    }
    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
    }
    pub fn channel_key(&self) -> &str {
        &self.credential
    }
    pub fn remote_device_id(&self) -> &str {
        &self.remote_device_id
    }
    pub fn remote_device_name(&self) -> &str {
        &self.remote_device_name
    }
    pub async fn heartbeat(&self) -> Result<bool, ErrorPayload> {
        Ok(matches!(
            self.request(Request::Heartbeat {
                version: PROTOCOL_VERSION,
                device_id: self.device_id.clone()
            })
            .await?,
            Response::Heartbeat
        ))
    }
    pub async fn capabilities(&self) -> Result<Value, ErrorPayload> {
        match self.request(Request::Capabilities).await? {
            Response::Capabilities { value } => Ok(value),
            response => Err(response_error(response)),
        }
    }
    pub async fn benchmark(
        &self,
        bytes: usize,
        samples: usize,
    ) -> Result<BenchmarkResult, ErrorPayload> {
        let bytes = bytes.clamp(4 * 1024, 512 * 1024);
        let samples = samples.clamp(1, 16);
        let payload = vec![0xa5; bytes];
        let mut latencies = Vec::with_capacity(samples);
        let started = Instant::now();
        for _ in 0..samples {
            let request_started = Instant::now();
            let (mut socket, mut noise) = self.connect_noise(false).await?;
            send_encrypted(
                &mut socket,
                &mut noise,
                &Request::Benchmark { size: bytes as u32 },
            )
            .await?;
            match receive_encrypted(&mut socket, &mut noise).await? {
                Response::Benchmark { size } if size as usize == bytes => {}
                response => return Err(response_error(response)),
            }
            super::channel::send_encrypted_bytes(&mut socket, &mut noise, &payload).await?;
            let echoed = super::channel::receive_encrypted_bytes(&mut socket, &mut noise).await?;
            if echoed.len() != bytes {
                return Err(ErrorPayload::new(
                    "benchmark_size",
                    "The peer benchmark payload did not match.",
                    None,
                ));
            }
            latencies.push(request_started.elapsed().as_secs_f64() * 1000.0);
        }
        let seconds = started.elapsed().as_secs_f64().max(0.000_001);
        latencies.sort_by(f64::total_cmp);
        let median = latencies[latencies.len() / 2];
        let p95 = latencies[((latencies.len() as f64 * 0.95).ceil() as usize)
            .saturating_sub(1)
            .min(latencies.len() - 1)];
        Ok(BenchmarkResult {
            throughput_mbps: (bytes * samples * 2 * 8) as f64 / seconds / 1_000_000.0,
            latency_median_ms: median,
            latency_p95_ms: p95,
            samples,
        })
    }
    pub async fn stop_worker(&self) -> Result<(), ErrorPayload> {
        match self.request(Request::StopWorker).await? {
            Response::WorkerStopped => Ok(()),
            response => Err(response_error(response)),
        }
    }
    pub async fn remote_models(&self) -> Result<Value, ErrorPayload> {
        match self.request(Request::Models).await? {
            Response::Models { models } => Ok(models),
            response => Err(response_error(response)),
        }
    }
    pub async fn proxy_chat(
        &self,
        messages: Value,
        settings: Value,
        images: Vec<String>,
    ) -> Result<String, ErrorPayload> {
        match self
            .request(Request::ProxyChat {
                messages,
                settings,
                images,
            })
            .await?
        {
            Response::ProxyChat { content } => Ok(content),
            response => Err(response_error(response)),
        }
    }
    pub async fn start_rpc_forwarder(self: &Arc<Self>) -> Result<RpcForwarder, ErrorPayload> {
        RpcForwarder::start(self.clone()).await
    }
    pub(crate) async fn open_rpc_stream(
        &self,
    ) -> Result<(TcpStream, snow::TransportState), ErrorPayload> {
        let (mut socket, mut noise) = self.connect_noise(false).await?;
        send_encrypted(&mut socket, &mut noise, &Request::RpcTunnel).await?;
        match receive_encrypted(&mut socket, &mut noise).await? {
            Response::RpcReady => Ok((socket, noise)),
            response => Err(response_error(response)),
        }
    }
    async fn request(&self, request: Request) -> Result<Response, ErrorPayload> {
        let pairing = matches!(request, Request::Pair { .. });
        let (mut socket, mut noise) = self.connect_noise(pairing).await?;
        send_encrypted(&mut socket, &mut noise, &request).await?;
        receive_encrypted(&mut socket, &mut noise).await
    }
    async fn connect_noise(
        &self,
        pairing: bool,
    ) -> Result<(TcpStream, snow::TransportState), ErrorPayload> {
        let mut socket =
            tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(self.endpoint))
                .await
                .map_err(|_| {
                    ErrorPayload::new(
                        "peer_timeout",
                        "The peer did not answer within five seconds.",
                        None,
                    )
                })?
                .map_err(protocol::io_error)?;
        let _ = socket.set_nodelay(true);
        protocol::write_plain(
            &mut socket,
            &ClientHello {
                device_id: self.device_id.clone(),
                pairing,
            },
        )
        .await?;
        let mut handshake = crypto::initiator(&self.credential)?;
        let mut outgoing = [0_u8; 1024];
        let count = handshake
            .write_message(&[], &mut outgoing)
            .map_err(crypto::noise_error)?;
        protocol::write_plain(&mut socket, &outgoing[..count].to_vec()).await?;
        let incoming: Vec<u8> = protocol::read_plain(&mut socket).await?;
        handshake
            .read_message(&incoming, &mut [])
            .map_err(crypto::noise_error)?;
        Ok((socket, crypto::transport(handshake)?))
    }
}

fn response_error(response: Response) -> ErrorPayload {
    match response {
        Response::Error { code, message } => ErrorPayload::new(code, message, None),
        _ => ErrorPayload::new(
            "peer_protocol",
            "The peer returned an unexpected response.",
            None,
        ),
    }
}
