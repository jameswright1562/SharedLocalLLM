use std::{
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpStream;

use super::{
    protocol::{self, ClientHello, Request, Response, PROTOCOL_VERSION},
    tunnel::RpcForwarder,
};
use crate::types::ErrorPayload;

#[derive(Clone)]
pub struct PeerClient {
    endpoint: SocketAddr,
    device_id: String,
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
    pub fn new(endpoint: SocketAddr, device_id: String) -> Self {
        Self {
            endpoint,
            device_id,
            remote_device_id: String::new(),
            remote_device_name: String::new(),
        }
    }

    pub async fn connect(
        &self,
        device_name: &str,
        capabilities: Value,
    ) -> Result<Self, ErrorPayload> {
        let mut client = self.clone();
        match client
            .request(Request::Connect {
                version: PROTOCOL_VERSION,
                device_id: self.device_id.clone(),
                device_name: device_name.into(),
                capabilities,
            })
            .await?
        {
            Response::Connected {
                device_id,
                device_name,
            } => {
                client.remote_device_id = device_id;
                client.remote_device_name = device_name;
                Ok(client)
            }
            response => Err(response_error(response)),
        }
    }

    pub fn endpoint(&self) -> SocketAddr {
        self.endpoint
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
            let mut socket = self.connect_plain().await?;
            protocol::write_plain(&mut socket, &Request::Benchmark { size: bytes as u32 }).await?;
            match protocol::read_plain(&mut socket).await? {
                Response::Benchmark { size } if size as usize == bytes => {}
                response => return Err(response_error(response)),
            }
            protocol::write_bytes(&mut socket, &payload).await?;
            let echoed = protocol::read_bytes(&mut socket).await?;
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
        RpcForwarder::start(self.clone(), false).await
    }

    pub(crate) async fn open_rpc_stream(&self) -> Result<TcpStream, ErrorPayload> {
        let mut socket = self.connect_plain().await?;
        protocol::write_plain(&mut socket, &Request::RpcTunnel).await?;
        match protocol::read_plain(&mut socket).await? {
            Response::RpcReady => Ok(socket),
            response => Err(response_error(response)),
        }
    }

    async fn request(&self, request: Request) -> Result<Response, ErrorPayload> {
        let mut socket = self.connect_plain().await?;
        protocol::write_plain(&mut socket, &request).await?;
        protocol::read_plain(&mut socket).await
    }

    async fn connect_plain(&self) -> Result<TcpStream, ErrorPayload> {
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
            },
        )
        .await?;
        Ok(socket)
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
