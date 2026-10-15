from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import time
from typing import Any, AsyncIterator

import psutil

from .errors import BackendError
from .peer_stream import request_stream, serve_events
from .store import save_peers, stored_peers

DISCOVERY_PORT = 49157
PEER_PORT = 49158
PROTOCOL_VERSION = 7
# Heartbeat-style ops must stay snappy. Generation ops have no response
# deadline because llama.cpp may spend an unbounded period ingesting a prompt
# without sending data; callers can still cancel generation explicitly.
PEER_REQUEST_TIMEOUT_SECONDS = 120
SLOW_PEER_OPS = {"chat", "benchmark_inference", "start_cluster"}


class RpcForwarder:
    def __init__(
        self, peer: "PeerManager", model_id: str | None = None, include_cpu: bool = False,
        peer_id: str | None = None,
    ) -> None:
        self.peer = peer
        self.model_id = model_id
        self.include_cpu = include_cpu
        self.peer_id = peer_id
        self.server: asyncio.AbstractServer | None = None
        self.endpoint: str | None = None
        # Raw tunnel bytes, counted on the coordinator side only. These feed
        # the load-progress transfer meter; they include RPC protocol
        # overhead, so they are an approximation, never an exact model share.
        self.bytes_to_worker = 0
        self.bytes_from_worker = 0

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
            host, port = self.peer.peer_endpoint(self.peer_id)
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
            await self._bridge_counted(reader, writer, remote_reader, remote_writer)
        except Exception as error:
            self.peer.runtime.store.log("WARN", "rpc_forwarder_failed", str(error))
        finally:
            writer.close()
            if remote_writer:
                remote_writer.close()

    async def _bridge_counted(
        self,
        local_reader: asyncio.StreamReader, local_writer: asyncio.StreamWriter,
        remote_reader: asyncio.StreamReader, remote_writer: asyncio.StreamWriter,
    ) -> None:
        async def copy(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter, attr: str
        ) -> None:
            try:
                while data := await reader.read(256 * 1024):
                    writer.write(data)
                    await writer.drain()
                    setattr(self, attr, getattr(self, attr) + len(data))
            finally:
                writer.close()

        await asyncio.gather(
            copy(local_reader, remote_writer, "bytes_to_worker"),
            copy(remote_reader, local_writer, "bytes_from_worker"),
            return_exceptions=True,
        )


class PeerManager:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.server: asyncio.AbstractServer | None = None
        self._broadcast_task: asyncio.Task[None] | None = None
        self.remote_models: list[dict[str, Any]] = []
        self.remote_models_by_peer: dict[str, list[dict[str, Any]]] = {}
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

    def peers(self) -> list[dict[str, Any]]:
        """All paired computers in stored order (up to N, e.g. 2 + 1 Mac)."""
        return stored_peers(self.runtime.store)

    def peer_by_id(self, peer_id: str) -> dict[str, Any] | None:
        return next((peer for peer in self.peers() if peer.get("id") == peer_id), None)

    def endpoint(self) -> tuple[str, int]:
        """First paired computer with an address (legacy single-peer callers)."""
        for peer in self.peers():
            if peer.get("address"):
                host, _, port = str(peer["address"]).partition(":")
                return host, int(port or PEER_PORT)
        raise BackendError("peer_unavailable", "Connect another computer first.")

    def peer_endpoint(self, peer_id: str | None) -> tuple[str, int]:
        """Address of one specific paired computer (multi-peer callers)."""
        if peer_id:
            peer = self.peer_by_id(peer_id)
            if peer and peer.get("address"):
                host, _, port = str(peer["address"]).partition(":")
                return host, int(port or PEER_PORT)
            raise BackendError(
                "peer_unavailable",
                "The selected computer is not paired or has no saved endpoint.",
                "Connect the computers again.",
            )
        return self.endpoint()

    def _save_peer_record(
        self, node: dict[str, Any], host: str, port: int,
        models: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Insert or replace one peer record by id without dropping the others,
        so adding a third computer (for example a Mac) keeps the second."""
        record = {
            "id": node["id"], "name": node["name"],
            "address": f"{host}:{port}", "capabilities": node,
        }
        peers = [peer for peer in self.peers() if peer.get("id") != node["id"]]
        peers.append(record)
        if len(peers) > 8:
            peers = peers[-8:]
        save_peers(self.runtime.store, peers)
        if models is not None:
            self.remote_models_by_peer[node["id"]] = list(models)
            self._rebuild_remote_models()
        return node

    def _rebuild_remote_models(self) -> None:
        order = [str(peer["id"]) for peer in self.peers() if peer.get("id") is not None]
        merged: list[dict[str, Any]] = []
        for peer_id in order:
            merged.extend(self.remote_models_by_peer.get(peer_id, []))
        for peer_id, models in self.remote_models_by_peer.items():
            if peer_id not in order:
                merged.extend(models)
        self.remote_models = merged

    def forget_peer_models(self, peer_id: str) -> None:
        self.remote_models_by_peer.pop(peer_id, None)
        self._rebuild_remote_models()

    async def connect(self, manual_endpoint: str | None) -> dict[str, Any]:
        if manual_endpoint:
            endpoints = [_parse_endpoint_rejecting_local(manual_endpoint)]
        else:
            discovered = await asyncio.to_thread(self._discover_sync, 4.0)
            known_ids = {peer.get("id") for peer in self.peers()}
            endpoints = [
                (host, port)
                for host, port, announcement in discovered
                if announcement.get("deviceId") not in known_ids
            ]
            if not endpoints:
                if self.peers():
                    raise BackendError(
                        "peer_already_connected",
                        "No new SharedLocalLLM computer was discovered.",
                        "Enter the third computer's IPv4 address to add it.",
                    )
                raise BackendError(
                    "peer_not_discovered", "No SharedLocalLLM computer was discovered.",
                    "Enter the other computer's Ethernet IPv4 address."
                )
        last_error: BackendError | None = None
        last_node: dict[str, Any] | None = None
        for host, port in endpoints:
            try:
                last_node = await self._connect_one(host, port)
            except BackendError as error:
                last_error = error
        if last_node is not None:
            return last_node
        raise last_error or BackendError("peer_connect_failed", "The connection attempt failed.")

    async def _connect_one(self, host: str, port: int) -> dict[str, Any]:
        response = await self.request_to(host, port, "connect", {
            "node": self.runtime.local_node,
            "models": self.runtime.local_models,
        })
        node = response["node"]
        node["online"] = True
        node["role"] = "worker"
        self._save_peer_record(node, host, port, response.get("models", []))
        return node

    async def request(
        self, op: str, data: dict[str, Any] | None = None, timeout: int | None = None
    ) -> Any:
        host, port = self.endpoint()
        if timeout is None:
            timeout = (
                None if op in SLOW_PEER_OPS else PEER_REQUEST_TIMEOUT_SECONDS
            )
        return await self.request_to(host, port, op, data, timeout)

    async def request_to(
        self, host: str, port: int, op: str, data: dict[str, Any] | None = None,
        timeout: int | None = PEER_REQUEST_TIMEOUT_SECONDS,
    ) -> Any:
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=2 * 1024 * 1024), 5
            )
            await _write_json(writer, {"version": PROTOCOL_VERSION, "op": op, "data": data or {}})
            response = (
                await _read_json(reader)
                if timeout is None
                else await asyncio.wait_for(_read_json(reader), timeout)
            )
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

    async def stream(
        self, op: str, data: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        host, port = self.endpoint()
        async for event in request_stream(host, port, PROTOCOL_VERSION, op, data or {}):
            yield event

    async def request_to_peer(
        self, peer_id: str, op: str, data: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> Any:
        """Send an op to one specific paired computer (3-node routing)."""
        host, port = self.peer_endpoint(peer_id)
        if timeout is None:
            timeout = None if op in SLOW_PEER_OPS else PEER_REQUEST_TIMEOUT_SECONDS
        return await self.request_to(host, port, op, data, timeout)

    async def stream_to_peer(
        self, peer_id: str, op: str, data: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        host, port = self.peer_endpoint(peer_id)
        async for event in request_stream(host, port, PROTOCOL_VERSION, op, data or {}):
            yield event

    async def heartbeat_peer(self, peer_id: str) -> dict[str, Any] | None:
        try:
            value = await self.request_to_peer(peer_id, "heartbeat")
            node = dict(value["node"])
            node["online"] = True
            node["role"] = "worker"
            peers = self.peers()
            for peer in peers:
                if peer.get("id") == peer_id and peer.get("capabilities") != node:
                    peer["capabilities"] = node
                    save_peers(self.runtime.store, peers)
                    break
            self.remote_models_by_peer[peer_id] = list(value.get("models", self.remote_models_by_peer.get(peer_id, [])))
            self._rebuild_remote_models()
            return node
        except BackendError:
            peers = self.peers()
            for peer in peers:
                capabilities = peer.get("capabilities")
                if peer.get("id") == peer_id and isinstance(capabilities, dict):
                    capabilities["online"] = False
                    save_peers(self.runtime.store, peers)
                    break
            return None

    async def heartbeat_all(self) -> dict[str, dict[str, Any]]:
        """Heartbeat every paired computer; returns the reachable subset by id."""
        found: dict[str, dict[str, Any]] = {}
        for peer in self.peers():
            node = await self.heartbeat_peer(str(peer.get("id")))
            if node is not None:
                found[str(peer.get("id"))] = node
        return found

    async def heartbeat(self) -> dict[str, Any] | None:
        """First reachable peer (legacy single-peer callers)."""
        for peer in self.peers():
            node = await self.heartbeat_peer(str(peer.get("id")))
            if node is not None:
                return node
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

    async def _reply(self, writer: asyncio.StreamWriter, payload: dict[str, Any]) -> bool:
        """Send a response, tolerating a peer that vanished before the answer.

        Long generations can outlive the caller; when the socket is already
        reset the write raises and would otherwise escape as noisy tracebacks.
        """
        try:
            await _write_json(writer, payload)
            return True
        except OSError as error:
            self.runtime.store.log("WARN", "peer_reply_undelivered", str(error))
            return False

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await _read_json(reader)
            if request.get("version") != PROTOCOL_VERSION:
                raise BackendError(
                    "peer_version", "The peer protocol versions do not match.",
                    "Run the Python-backend branch on both computers."
                )
            op = request.get("op")
            if not isinstance(op, str):
                raise BackendError("peer_request_invalid", "The peer operation must be a string.")
            raw_data = request.get("data")
            data = raw_data if isinstance(raw_data, dict) else {}
            if op == "rpc_tunnel":
                include_cpu = bool(data.get("includeCpu", False))
                model_id = str(data.get("modelId") or "") or None
                rpc_reader, rpc_writer = await self.runtime.inference.open_rpc_worker_connection(include_cpu)
                self._rpc_tunnels += 1
                coordinator = self._coordinator_for_source(writer)
                self.runtime.cluster = {
                    "status": "running",
                    "coordinatorNodeId": coordinator,
                    "workerNodeId": self.runtime.local_node["id"],
                    "modelId": model_id,
                }
                self.runtime._publish_local_cluster()
                try:
                    if not await self._reply(writer, {"ok": True}):
                        return
                    await _bridge(reader, writer, rpc_reader, rpc_writer)
                finally:
                    self._rpc_tunnels = max(0, self._rpc_tunnels - 1)
                    if self._rpc_tunnels == 0:
                        self.runtime.cluster = {
                            "status": "ready" if self.peers() else "idle"
                        }
                        self.runtime._publish_local_cluster()
                return
            if op in ("chat_stream", "chat_openai_stream"):
                await serve_events(
                    reader,
                    writer,
                    self._dispatch_stream(op, data),
                )
                return
            if op == "connect":
                self._accept_peer(data, writer)
            value = await self._dispatch(op, data)
            await self._reply(writer, {"ok": True, "value": value})
        except BackendError as error:
            await self._reply(writer, {"ok": False, **error.to_dict()})
        except Exception as error:
            await self._reply(writer, {"ok": False, "code": "peer_internal", "message": str(error)})
        finally:
            writer.close()

    def _coordinator_for_source(self, writer: asyncio.StreamWriter) -> str | None:
        """Id of the paired computer that opened this connection, if known.

        Matches by source IP so a 3-node worker attributes the cluster to the
        coordinator that actually connected, not just the first stored peer.
        """
        source = writer.get_extra_info("peername")
        source_ip = source[0] if source else None
        for peer in self.peers():
            if source_ip and str(peer.get("address", "")).startswith(f"{source_ip}:"):
                return str(peer.get("id"))
        peers = self.peers()
        return str(peers[0].get("id")) if peers else None

    def _accept_peer(self, data: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        node = data.get("node")
        source = writer.get_extra_info("peername")
        if not isinstance(node, dict) or not source:
            raise BackendError("peer_connect_invalid", "The peer supplied invalid capabilities.")
        remote = dict(node)
        remote["online"] = True
        remote["role"] = "worker"
        self._save_peer_record(remote, source[0], PEER_PORT, list(data.get("models") or []))
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
            return await self.runtime.chat(
                data["messages"], data["settings"], data.get("images", []),
                proxy_peer=False, tools=data.get("tools"), tool_choice=data.get("toolChoice"),
            )
        if op == "cancel_generation":
            return self.runtime.inference.cancel()
        if op == "benchmark_inference":
            return await self.runtime.run_inference_benchmark(data["modelId"], proxy_peer=False)
        raise BackendError("peer_request_unknown", f"Unknown peer operation: {op}")

    async def _dispatch_stream(
        self, op: str, data: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        if op == "chat_stream":
            async for event in self.runtime.chat_stream_events(
                data["messages"], data["settings"], data.get("images", []),
                proxy_peer=False,
            ):
                yield event
            return
        if op == "chat_openai_stream":
            async for event in self.runtime.chat_openai_stream(
                data["messages"], data["settings"], data.get("tools"),
                data.get("toolChoice"), proxy_peer=False,
            ):
                yield event
            return
        raise BackendError("peer_request_unknown", f"Unknown peer stream operation: {op}")

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


def _parse_endpoint_rejecting_local(value: str) -> tuple[str, int]:
    """Parse a manual endpoint, refusing addresses owned by this computer."""
    host, port = _parse_endpoint(value)
    try:
        local = {
            address.address
            for addresses in psutil.net_if_addrs().values()
            for address in addresses
            if address.family == socket.AF_INET and address.address
        }
    except OSError:
        local = set()
    if host in local:
        raise BackendError(
            "peer_endpoint_is_local",
            f"{host} belongs to this computer, not another computer.",
            "Enter the other computer's IPv4 address.",
        )
    return host, port
