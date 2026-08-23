from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import time
from typing import Any

import psutil

from .errors import BackendError

DISCOVERY_PORT = 49157
PEER_PORT = 49158
PROTOCOL_VERSION = 5


class RpcForwarder:
    def __init__(
        self, peer: "PeerManager", model_id: str, include_cpu: bool = False
    ) -> None:
        self.peer = peer
        self.model_id = model_id
        self.include_cpu = include_cpu
        self.server: asyncio.AbstractServer | None = None
        self.endpoint: str | None = None

    async def start(self) -> str:
        if self.server and self.endpoint:
            return self.endpoint
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = int(self.server.sockets[0].getsockname()[1])
        self.endpoint = f"127.0.0.1:{port}"
        self.peer.runtime.store.log("INFO", "rpc_forwarder_ready", self.endpoint)
        return self.endpoint

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.server = None
        self.endpoint = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        remote_writer: asyncio.StreamWriter | None = None
        try:
            host, port = self.peer.endpoint()
            remote_reader, remote_writer = await asyncio.open_connection(
                host, port, limit=2 * 1024 * 1024
            )
            await _write_json(remote_writer, {
                "version": PROTOCOL_VERSION,
                "op": "rpc_tunnel",
                "data": {"includeCpu": self.include_cpu, "modelId": self.model_id},
            })
            ready = await _read_json(remote_reader)
            if not ready.get("ok"):
                raise BackendError("rpc_tunnel_failed", ready.get("message", "RPC tunnel failed"))
            await _bridge(reader, writer, remote_reader, remote_writer)
        except Exception as error:
            self.peer.runtime.store.log("WARN", "rpc_forwarder_failed", str(error))
        finally:
            writer.close()
            if remote_writer:
                remote_writer.close()


class PeerManager:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.server: asyncio.AbstractServer | None = None
        self._broadcast_task: asyncio.Task[None] | None = None
        self.remote_models: list[dict[str, Any]] = []
        self._rpc_tunnels = 0

    async def start(self) -> None:
        if self.server:
            return
        self.server = await asyncio.start_server(
            self._handle_connection, "0.0.0.0", PEER_PORT, limit=2 * 1024 * 1024
        )
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self.runtime.store.log("INFO", "peer_listener_ready", f"0.0.0.0:{PEER_PORT}")

    async def stop(self) -> None:
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.server = None

    def endpoint(self) -> tuple[str, int]:
        peer = self.runtime.store.get("peer")
        if not peer or not peer.get("address"):
            raise BackendError("peer_unavailable", "Connect another computer first.")
        host, _, port = peer["address"].partition(":")
        return host, int(port or PEER_PORT)

    async def connect(self, manual_endpoint: str | None) -> dict[str, Any]:
        if manual_endpoint:
            host, port = _parse_endpoint(manual_endpoint)
        else:
            discovered = await asyncio.to_thread(self._discover_sync, 4.0)
            if not discovered:
                raise BackendError(
                    "peer_not_discovered", "No SharedLocalLLM computer was discovered.",
                    "Enter the other computer's Ethernet IPv4 address."
                )
            host, port, _ = discovered[0]
        response = await self.request_to(host, port, "connect", {
            "node": self.runtime.local_node,
            "models": self.runtime.local_models,
        })
        node = response["node"]
        node["online"] = True
        node["role"] = "worker"
        self.remote_models = response.get("models", [])
        self.runtime.store.update(peer={
            "id": node["id"], "name": node["name"], "address": f"{host}:{port}", "capabilities": node,
        })
        return node

    async def request(self, op: str, data: dict[str, Any] | None = None) -> Any:
        host, port = self.endpoint()
        return await self.request_to(host, port, op, data)

    async def request_to(self, host: str, port: int, op: str, data: dict[str, Any] | None = None) -> Any:
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=2 * 1024 * 1024), 5
            )
            await _write_json(writer, {"version": PROTOCOL_VERSION, "op": op, "data": data or {}})
            response = await asyncio.wait_for(_read_json(reader), 120)
        except (OSError, asyncio.TimeoutError, ValueError) as error:
            raise BackendError("peer_unavailable", f"The other computer did not answer: {error}") from error
        finally:
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
        if not response.get("ok"):
            raise BackendError(
                response.get("code", "peer_error"), response.get("message", "Peer request failed"), response.get("action")
            )
        return response.get("value", {})

    async def heartbeat(self) -> dict[str, Any] | None:
        try:
            value = await self.request("heartbeat")
            node = dict(value["node"])
            node["online"] = True
            node["role"] = "worker"
            peer = self.runtime.store.get("peer")
            if peer and peer.get("capabilities") != node:
                peer["capabilities"] = node
                self.runtime.store.update(peer=peer)
            self.remote_models = value.get("models", self.remote_models)
            return node
        except BackendError:
            return None

    async def network_benchmark(self) -> dict[str, Any]:
        samples: list[float] = []
        failures = 0
        for _ in range(5):
            started = time.perf_counter()
            try:
                await self.request("heartbeat")
                samples.append((time.perf_counter() - started) * 1000)
            except BackendError:
                failures += 1
        if not samples:
            raise BackendError("network_test_failed", "Every peer benchmark request failed.")
        payload = "x" * (256 * 1024)
        started = time.perf_counter()
        uploaded = await self.request("upload", {"payload": payload})
        upload_elapsed = max(time.perf_counter() - started, 0.001)
        upload_bits = int(uploaded.get("size", 0)) * 8
        started = time.perf_counter()
        downloaded = await self.request("download", {"size": len(payload)})
        download_elapsed = max(time.perf_counter() - started, 0.001)
        download_bits = len(downloaded.get("payload", "")) * 8
        samples.sort()
        median = samples[len(samples) // 2]
        p95 = samples[-1]
        down = download_bits / download_elapsed / 1_000_000
        up = upload_bits / upload_elapsed / 1_000_000
        classification = "good" if median < 10 and min(down, up) >= 50 else "usable" if median < 30 else "poor"
        return {
            "downMbps": round(down, 2), "upMbps": round(up, 2),
            "latencyMedianMs": round(median, 2), "latencyP95Ms": round(p95, 2),
            "jitterMs": round(max(samples) - min(samples), 2),
            "packetLossPercent": round(failures / 5 * 100, 1),
            "classification": classification, "adapter": self.runtime.local_node["adapter"]["name"],
        }

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await _read_json(reader)
            if request.get("version") != PROTOCOL_VERSION:
                raise BackendError(
                    "peer_version", "The peer protocol versions do not match.",
                    "Run the Python-backend branch on both computers."
                )
            op = request.get("op")
            data = request.get("data") or {}
            if op == "rpc_tunnel":
                include_cpu = bool(data.get("includeCpu", False))
                model_id = str(data.get("modelId") or "") or None
                rpc_reader, rpc_writer = await self.runtime.inference.open_rpc_worker_connection(include_cpu)
                self._rpc_tunnels += 1
                self.runtime.cluster = {
                    "status": "running",
                    "coordinatorNodeId": (self.runtime.store.get("peer") or {}).get("id"),
                    "workerNodeId": self.runtime.local_node["id"],
                    "modelId": model_id,
                }
                self.runtime._publish_local_cluster()
                try:
                    await _write_json(writer, {"ok": True})
                    await _bridge(reader, writer, rpc_reader, rpc_writer)
                finally:
                    self._rpc_tunnels = max(0, self._rpc_tunnels - 1)
                    if self._rpc_tunnels == 0:
                        self.runtime.cluster = {
                            "status": "ready" if self.runtime.store.get("peer") else "idle"
                        }
                        self.runtime._publish_local_cluster()
                return
            if op == "connect":
                self._accept_peer(data, writer)
            value = await self._dispatch(op, data)
            await _write_json(writer, {"ok": True, "value": value})
        except BackendError as error:
            await _write_json(writer, {"ok": False, **error.to_dict()})
        except Exception as error:
            await _write_json(writer, {"ok": False, "code": "peer_internal", "message": str(error)})
        finally:
            writer.close()

    def _accept_peer(self, data: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        node = data.get("node")
        source = writer.get_extra_info("peername")
        if not isinstance(node, dict) or not source:
            raise BackendError("peer_connect_invalid", "The peer supplied invalid capabilities.")
        remote = dict(node)
        remote["online"] = True
        remote["role"] = "worker"
        self.remote_models = list(data.get("models") or [])
        self.runtime.store.update(peer={
            "id": remote["id"], "name": remote["name"],
            "address": f"{source[0]}:{PEER_PORT}", "capabilities": remote,
        })
        self.runtime._merge_models()

    async def _dispatch(self, op: str, data: dict[str, Any]) -> Any:
        if op in ("connect", "heartbeat"):
            return {"node": self.runtime.local_node, "models": self.runtime.local_models}
        if op == "models":
            return {"models": self.runtime.local_models}
        if op == "upload":
            return {"size": len(str(data.get("payload", "")))}
        if op == "download":
            size = max(4096, min(512 * 1024, int(data.get("size", 0))))
            return {"payload": "x" * size}
        if op == "start_cluster":
            return await self.runtime.start_cluster(data["modelId"], data["loadConfig"], allow_peer=False)
        if op == "stop_cluster":
            return await self.runtime.stop_cluster(proxy_peer=False)
        if op == "chat":
            return await self.runtime.chat(data["messages"], data["settings"], data.get("images", []), proxy_peer=False)
        if op == "benchmark_inference":
            return await self.runtime.run_inference_benchmark(data["modelId"], proxy_peer=False)
        raise BackendError("peer_request_unknown", f"Unknown peer operation: {op}")

    async def _broadcast_loop(self) -> None:
        sockets = _broadcast_sockets()
        try:
            while True:
                payload = json.dumps({
                    "protocolVersion": PROTOCOL_VERSION,
                    "deviceId": self.runtime.local_node["id"],
                    "deviceName": self.runtime.local_node["name"],
                    "peerPort": PEER_PORT,
                }).encode()
                for sock, target in sockets:
                    try:
                        sock.sendto(payload, target)
                    except OSError:
                        pass
                await asyncio.sleep(1)
        finally:
            for sock, _ in sockets:
                sock.close()

    def _discover_sync(self, seconds: float) -> list[tuple[str, int, dict[str, Any]]]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", DISCOVERY_PORT))
        sock.settimeout(0.25)
        found: dict[str, tuple[str, int, dict[str, Any]]] = {}
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                payload, source = sock.recvfrom(64 * 1024)
                value = json.loads(payload)
                if value.get("protocolVersion") == PROTOCOL_VERSION and value.get("deviceId") != self.runtime.local_node["id"]:
                    found[value["deviceId"]] = (source[0], int(value.get("peerPort", PEER_PORT)), value)
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        sock.close()
        return list(found.values())


async def _read_json(reader: asyncio.StreamReader) -> dict[str, Any]:
    line = await reader.readline()
    if not line:
        raise BackendError("peer_closed", "Peer closed the connection.")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise BackendError("peer_protocol", "The peer returned invalid JSON.") from error
    if not isinstance(value, dict):
        raise BackendError("peer_protocol", "The peer response must be a JSON object.")
    return value


async def _write_json(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
    writer.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")
    await writer.drain()


async def _bridge(
    left_reader: asyncio.StreamReader, left_writer: asyncio.StreamWriter,
    right_reader: asyncio.StreamReader, right_writer: asyncio.StreamWriter,
) -> None:
    async def copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(256 * 1024):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    await asyncio.gather(copy(left_reader, right_writer), copy(right_reader, left_writer), return_exceptions=True)


def _broadcast_sockets() -> list[tuple[socket.socket, tuple[str, int]]]:
    values: list[tuple[socket.socket, tuple[str, int]]] = []
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family != socket.AF_INET or not address.address or address.address.startswith("127."):
                continue
            broadcast = address.broadcast or "255.255.255.255"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.bind((address.address, 0))
                values.append((sock, (broadcast, DISCOVERY_PORT)))
            except OSError:
                continue
    if not values:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        values.append((sock, ("255.255.255.255", DISCOVERY_PORT)))
    return values


def _parse_endpoint(value: str) -> tuple[str, int]:
    text = value.strip().replace("http://", "").replace("https://", "").rstrip("/")
    host, sep, port = text.partition(":")
    try:
        ipaddress.ip_address(host)
    except ValueError as error:
        raise BackendError("peer_endpoint_invalid", "Enter the other computer's IPv4 address.") from error
    try:
        parsed_port = int(port) if sep else PEER_PORT
    except ValueError as error:
        raise BackendError("peer_endpoint_invalid", "Enter a numeric peer port.") from error
    if not 1 <= parsed_port <= 65535:
        raise BackendError("peer_endpoint_invalid", "Peer port must be between 1 and 65535.")
    return host, parsed_port
