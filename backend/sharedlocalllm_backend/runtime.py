from __future__ import annotations

import asyncio
import inspect
import json
import secrets
import socket
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from .errors import BackendError
from .hardware import probe_node
from .inference import InferenceEngine, layer_totals
from .models import discover_local, lm_studio_roots, merge_remote, refresh_fits
from .peer import PeerManager, RpcForwarder
from .placement import estimate_split, normalize_load_config, validate_fit
from .rpc_native import runtime_health
from .server_engine import ServerEngine
from .store import Store


class BackendRuntime:
    def __init__(self) -> None:
        self.store = Store()
        self.local_node = probe_node(
            self.store.get("installId"), self.store.get("deviceName"), "coordinator"
        )
        self.local_models: list[dict[str, Any]] = []
        self.model_paths: dict[str, str] = {}
        self.models: list[dict[str, Any]] = []
        self.network: dict[str, Any] | None = None
        self.cluster: dict[str, Any] = {"status": "idle"}
        self.peer_active_model_id: str | None = None
        self.inference = InferenceEngine(self.store)
        self.peer = PeerManager(self)
        self.server_engine = ServerEngine(self.store)
        self._server_forwarder: RpcForwarder | None = None
        self.api_port_changed: Callable[[int], Awaitable[None]] | None = None
        self.api_health: Callable[[], bool] | None = None
        self._peer_refresh_task: asyncio.Task[None] | None = None
        self._runtime = runtime_health()
        self.discover_models()

    async def start(self) -> None:
        await self.peer.start()
        await self.refresh_peer()
        self._peer_refresh_task = asyncio.create_task(self._peer_refresh_loop())

    async def stop(self) -> None:
        if self._peer_refresh_task:
            self._peer_refresh_task.cancel()
            try:
                await self._peer_refresh_task
            except asyncio.CancelledError:
                pass
            self._peer_refresh_task = None
        await self.inference.unload()
        await self.server_engine.stop()
        await self._stop_server_forwarder()
        await self.peer.stop()

    async def _peer_refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(8)
            try:
                await self.refresh_peer()
            except Exception as error:
                # One malformed peer record or bad stored address must not kill
                # heartbeats for the process lifetime; log and keep retrying.
                self.store.log(
                    "WARN", "peer_refresh_failed",
                    f"The worker heartbeat failed; retrying in 8 seconds: {error}",
                )

    async def refresh_peer(self) -> None:
        peer = self.store.get("peer")
        if not peer:
            self.peer_active_model_id = None
            self._merge_models()
            return
        node = await self.peer.heartbeat()
        if node:
            node["online"] = True
            node["role"] = "worker"
            self.peer_active_model_id = (
                node.get("clusterModelId") if node.get("clusterStatus") == "running" else None
            )
        elif peer.get("capabilities", {}).get("online", True):
            peer["capabilities"]["online"] = False
            self.peer_active_model_id = None
            self.store.update(peer=peer)
        refresh_fits(self.local_models, self.local_node, self._peer_node())
        self._merge_models()

    def discover_models(self) -> list[dict[str, Any]]:
        roots = lm_studio_roots() + [Path(value) for value in self.store.get("customModelDirectories", [])]
        peer = self._peer_node()
        self.local_models, self.model_paths = discover_local(
            roots, self.local_node["id"], self.local_node, peer
        )
        self._merge_models()
        return self.models

    def _merge_models(self) -> None:
        self.models = merge_remote(self.local_models, self.peer.remote_models)

    def _peer_node(self) -> dict[str, Any] | None:
        peer = self.store.get("peer")
        return peer.get("capabilities") if peer else None

    def _cluster_nodes(self) -> list[dict[str, Any]]:
        nodes = [self.local_node]
        peer = self._peer_node()
        if peer and peer.get("online", True):
            nodes.append(peer)
        return nodes

    def snapshot(self) -> dict[str, Any]:
        nodes = [self.local_node]
        peer = self._peer_node()
        if peer:
            nodes.append(peer)
        return {
            "setupComplete": bool(self.store.get("setupComplete")),
            "runtime": dict(self._runtime),
            "deviceName": self.local_node["name"],
            "apiPort": int(self.store.get("apiPort", 11435)),
            "authRequired": bool(self.store.get("authRequired", True)),
            "autostart": bool(self.store.get("autostart")),
            "nodes": nodes,
            "models": self.models,
            "modelLoadConfigs": self.store.model_load_configs(),
            "modelDirectories": self._directories(),
            "network": self.network,
            "cluster": self.cluster,
            "benchmarks": self.store.get("benchmarks", []),
            "logs": self.store.logs(),
        }

    async def dispatch(self, command: str, args: dict[str, Any]) -> Any:
        handlers = {
            "get_app_snapshot": lambda: self.snapshot(),
            "complete_setup": lambda: self.complete_setup(args["deviceName"]),
            "update_settings": lambda: self.update_settings(args["settings"]),
            "install_runtime": self.ensure_runtime,
            "refresh_hardware": self.refresh_hardware,
            "discover_models": self.discover_models,
            "add_model_directory": lambda: self.add_model_directory(args["path"]),
            "remove_model_directory": lambda: self.remove_model_directory(args["id"]),
            "run_network_test": self.run_network_test,
            "connect_peer": lambda: self.connect_peer(args.get("manualEndpoint")),
            "reset_pairing": self.reset_pairing,
            "estimate_model_split": lambda: self.estimate_model_split(args["modelId"], args["loadConfig"]),
            "start_cluster": lambda: self.start_cluster(args["modelId"], args["loadConfig"]),
            "stop_cluster": self.stop_cluster,
            "run_inference_benchmark": lambda: self.run_inference_benchmark(args["modelId"]),
            "cancel_inference_benchmark": self.cancel_generation,
            "send_chat_message": lambda: self.chat(args["messages"], args["settings"], args.get("images", [])),
            "cancel_generation": self.cancel_generation,
            "get_api_config": self.get_api_config,
            "regenerate_api_key": self.regenerate_api_key,
            "try_api_request": self.try_api_request,
        }
        handler = handlers.get(command)
        if not handler:
            raise BackendError("command_unknown", f"Unknown backend command: {command}")
        value = handler()
        if inspect.isawaitable(value):
            return await value
        return value

    def ensure_runtime(self) -> dict[str, Any]:
        self._runtime = runtime_health()
        if self._runtime["status"] != "ready":
            raise BackendError(
                "runtime_unavailable",
                str(self._runtime.get("error") or "The llama.cpp runtime is unavailable."),
                "Run pnpm backend:install in development, or reinstall the desktop application.",
            )
        return self.snapshot()

    def complete_setup(self, device_name: str) -> dict[str, Any]:
        name = device_name.strip()
        if not name or len(name) > 80:
            raise BackendError("device_name_invalid", "Device name must contain between 1 and 80 characters.")
        self.local_node["name"] = name
        self.store.update(deviceName=name, setupComplete=True)
        return self.snapshot()

    async def update_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        name = str(settings.get("deviceName", "")).strip()
        try:
            port = int(settings.get("apiPort", 11435))
        except (TypeError, ValueError) as error:
            raise BackendError("settings_invalid", "API port must be a whole number.") from error
        if not name or len(name) > 80 or not 1024 <= port <= 65535 or port == 11436:
            raise BackendError("settings_invalid", "Use a valid device name and API port from 1024-65535.")
        old_port = int(self.store.get("apiPort", 11435))
        if port != old_port and not _port_available(port):
            raise BackendError(
                "api_port_in_use", f"127.0.0.1:{port} is already in use.", "Choose another local API port."
            )
        if port != old_port and self.api_port_changed:
            await self.api_port_changed(port)
        self.local_node["name"] = name
        self.store.update(
            deviceName=name,
            apiPort=port,
            autostart=bool(settings.get("autostart")),
            authRequired=bool(settings.get("authRequired", True)),
        )
        return self.snapshot()

    def refresh_hardware(self) -> dict[str, Any]:
        status = self.local_node.get("clusterStatus")
        model_id = self.local_node.get("clusterModelId")
        self.local_node = probe_node(
            self.store.get("installId"), self.store.get("deviceName"), "coordinator"
        )
        if status:
            self.local_node["clusterStatus"] = status
        if model_id:
            self.local_node["clusterModelId"] = model_id
        self.discover_models()
        return self.snapshot()

    def _directories(self) -> list[dict[str, Any]]:
        values = []
        for path in lm_studio_roots():
            values.append({"id": f"lm:{path}", "nodeId": self.local_node["id"], "path": str(path), "source": "lm-studio"})
        for path in self.store.get("customModelDirectories", []):
            values.append({"id": f"custom:{path}", "nodeId": self.local_node["id"], "path": path, "source": "custom"})
        return values

    def add_model_directory(self, path: str) -> dict[str, Any] | None:
        candidate = Path(path)
        if not candidate.is_dir():
            raise BackendError("model_directory_invalid", "Choose an existing model directory.")
        current = list(self.store.get("customModelDirectories", []))
        value = str(candidate.resolve())
        if value not in current:
            current.append(value)
            self.store.update(customModelDirectories=current)
        self.discover_models()
        return {"id": f"custom:{value}", "nodeId": self.local_node["id"], "path": value, "source": "custom"}

    def remove_model_directory(self, directory_id: str) -> None:
        current = list(self.store.get("customModelDirectories", []))
        if directory_id.startswith("custom:"):
            target = directory_id[len("custom:"):]
            current = [value for value in current if value != target]
            self.store.update(customModelDirectories=current)
            self.discover_models()

    async def connect_peer(self, manual_endpoint: str | None) -> dict[str, Any]:
        node = await self.peer.connect(manual_endpoint)
        node["role"] = "worker"
        refresh_fits(self.local_models, self.local_node, node)
        self._merge_models()
        return node

    async def reset_pairing(self) -> dict[str, Any]:
        if self.cluster.get("status") in ("loading", "running", "error"):
            try:
                await self.stop_cluster()
            except BackendError:
                await self.inference.unload()
                await self.server_engine.stop()
                await self._stop_server_forwarder()
        self.store.update(peer=None)
        self.peer.remote_models = []
        self.peer_active_model_id = None
        self.network = None
        self.cluster = {"status": "idle"}
        self._publish_local_cluster()
        refresh_fits(self.local_models, self.local_node, None)
        self._merge_models()
        return self.snapshot()

    async def run_network_test(self) -> dict[str, Any]:
        self.network = await self.peer.network_benchmark()
        return self.network

    def estimate_model_split(self, model_id: str, config: dict[str, Any]) -> dict[str, Any]:
        model = self._model(model_id)
        nodes = self._cluster_nodes()
        normalized = normalize_load_config(model, config, nodes)
        return estimate_split(model, normalized, nodes)

    async def start_cluster(
        self, model_id: str, load_config: dict[str, Any], allow_peer: bool = True,
        save_config: bool = True,
    ) -> dict[str, Any]:
        model = self._model(model_id)
        peer = self.store.get("peer")
        if model.get("remoteOnly"):
            if not allow_peer or not peer:
                raise BackendError("model_not_local", "This model is stored only on the other computer.")
            await self.peer.request("start_cluster", {"modelId": model_id, "loadConfig": load_config})
            if save_config:
                self.store.save_model_load_config(model_id, load_config)
            self.cluster = {
                "status": "running", "coordinatorNodeId": peer["id"],
                "workerNodeId": self.local_node["id"], "modelId": model_id,
            }
            self._publish_local_cluster()
            return self.cluster
        path = self.model_paths.get(model_id)
        if not path:
            raise BackendError("model_not_found", "The selected local GGUF is unavailable.")
        nodes = self._cluster_nodes()
        normalized_config = normalize_load_config(model, load_config, nodes)
        if model.get("layerCount"):
            validate_fit(
                estimate_split(model, normalized_config, nodes),
                bool(normalized_config.get("force")),
            )
        peer_id = peer["id"] if peer else None
        engine = str(
            load_config.get("engine")
            or ("llama-server" if model.get("mtp") else "builtin")
        )
        if engine == "llama-server":
            exe = self.server_engine.available()
            if exe is None:
                self.store.log(
                    "WARN", "llama_server_missing",
                    "Pinned llama-server is not installed; using the built-in engine.",
                )
            else:
                await self.inference.unload()
                await self._stop_server_forwarder()
                include_remote_cpu = bool(normalized_config.get("includeRemoteCpu"))
                remote_gpu, remote_cpu, _local_layers = layer_totals(
                    normalized_config.get("gpuLayers") or [], peer_id,
                    self.local_node["id"], include_remote_cpu,
                )
                forwarder: RpcForwarder | None = None
                rpc_endpoint = None
                if peer_id and (remote_gpu + remote_cpu > 0 or model.get("fit") == "combined-gpu"):
                    # Same tunnel the built-in engine uses: llama-server reaches
                    # the worker's RPC daemon through loopback only.
                    forwarder = RpcForwarder(self.peer, model_id=model_id, include_cpu=remote_cpu > 0)
                    rpc_endpoint = await forwarder.start()
                self.cluster = {
                    "status": "loading", "coordinatorNodeId": self.local_node["id"],
                    "modelId": model_id,
                }
                self._publish_local_cluster()
                try:
                    await self.server_engine.start(
                        exe=exe, model_path=path, model_id=model_id,
                        context=int(normalized_config.get("contextSize", 4096)),
                        api_key=self.store.get("apiKey"), mtp=bool(model.get("mtp")),
                        rpc_endpoint=rpc_endpoint,
                        load_config=normalized_config,
                    )
                except Exception:
                    await self.server_engine.stop()
                    if forwarder:
                        await forwarder.stop()
                    self.cluster = {
                        "status": "error", "coordinatorNodeId": self.local_node["id"],
                        "modelId": model_id, "error": "llama-server failed to start",
                    }
                    self._publish_local_cluster()
                    raise
                self._server_forwarder = forwarder
                if save_config:
                    self.store.save_model_load_config(model_id, normalized_config)
                self.cluster = {
                    "status": "running", "coordinatorNodeId": self.local_node["id"],
                    "modelId": model_id, "engine": "llama-server",
                }
                self._publish_local_cluster()
                return self.cluster
        await self.server_engine.stop()
        await self._stop_server_forwarder()
        self.cluster = {"status": "loading", "coordinatorNodeId": self.local_node["id"], "modelId": model_id}
        self._publish_local_cluster()
        try:
            await self.inference.load(
                model, path, normalized_config, self.peer, self.local_node["id"], peer_id
            )
            if save_config:
                self.store.save_model_load_config(model_id, normalized_config)
            uses_peer = bool(peer_id and any(
                x.get("nodeId") == peer_id and x.get("layers", 0) > 0
                for x in normalized_config.get("gpuLayers", [])
            ))
            self.cluster = {
                "status": "running", "coordinatorNodeId": self.local_node["id"],
                "workerNodeId": peer_id if uses_peer else None, "modelId": model_id,
            }
            self._publish_local_cluster()
            return self.cluster
        except Exception as error:
            await self.inference.unload()
            self.cluster = {
                "status": "error", "coordinatorNodeId": self.local_node["id"],
                "modelId": model_id, "error": str(error),
            }
            self._publish_local_cluster()
            if isinstance(error, BackendError):
                raise
            raise BackendError("model_load_failed", str(error)) from error

    def _remote_coordinator(self) -> str | None:
        """Id of the other computer that is running this computer's model.

        Prefers the locally recorded cluster session (this computer joined a
        peer-launched cluster). When no local session exists, trusts the peer
        heartbeat: a model loaded entirely on the other computer never opens an
        RPC tunnel here, so its heartbeat is the only signal available.
        """
        if self.cluster.get("status") in ("loading", "running", "stopping"):
            coordinator = self.cluster.get("coordinatorNodeId")
            if coordinator and coordinator != self.local_node["id"]:
                return str(coordinator)
            return None
        peer = self.store.get("peer")
        if peer and self.peer_active_model_id:
            return str(peer["id"])
        return None

    async def _stop_server_forwarder(self) -> None:
        forwarder, self._server_forwarder = self._server_forwarder, None
        if forwarder:
            await forwarder.stop()

    async def stop_cluster(self, proxy_peer: bool = True) -> dict[str, Any]:
        remote_coordinator = self._remote_coordinator() if proxy_peer else None
        if remote_coordinator and self.store.get("peer"):
            try:
                await self.peer.request("stop_cluster")
            except BackendError as error:
                self.cluster = {**self.cluster, "status": "error", "error": error.message}
                self._publish_local_cluster()
                raise
        await self.inference.unload()
        await self.server_engine.stop()
        await self._stop_server_forwarder()
        self.cluster = {"status": "ready" if self.store.get("peer") else "idle"}
        self._publish_local_cluster()
        return self.cluster

    def _publish_local_cluster(self) -> None:
        self.local_node["clusterStatus"] = self.cluster.get("status")
        model_id = self.cluster.get("modelId")
        if model_id:
            self.local_node["clusterModelId"] = model_id
        else:
            self.local_node.pop("clusterModelId", None)

    async def chat(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str],
        proxy_peer: bool = True, tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict[str, Any]:
        if proxy_peer and self._remote_coordinator():
            payload: dict[str, Any] = {
                "messages": messages, "settings": settings, "images": images,
                "tools": tools, "toolChoice": tool_choice,
            }
            return await self.peer.request("chat", payload)
        if self.server_engine.active:
            if images:
                raise BackendError(
                    "vision_not_migrated",
                    "Vision attachments are not enabled in the Python migration yet.",
                )
            return await self.server_engine.chat(messages, settings, tools, tool_choice)
        return await self.inference.chat(messages, settings, images, tools, tool_choice)

    async def chat_stream_events(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str]
    ) -> AsyncIterator[dict[str, Any]]:
        if self._remote_coordinator():
            result = await self.chat(messages, settings, images)
            if result.get("reasoning"):
                yield {"type": "reasoning", "content": result["reasoning"]}
            if result.get("content"):
                yield {"type": "token", "content": result["content"]}
            if result.get("tokensPerSecond"):
                yield {"type": "stats", "tokensPerSecond": result["tokensPerSecond"]}
            yield {"type": "done"}
            return
        if self.server_engine.active:
            if images:
                raise BackendError(
                    "vision_not_migrated",
                    "Vision attachments are not enabled in the Python migration yet.",
                )
            async for event in self.server_engine.chat_stream(messages, settings):
                yield event
            return
        async for event in self.inference.chat_stream(messages, settings, images):
            yield event

    async def cancel_generation(self) -> None:
        if self._remote_coordinator():
            try:
                await self.peer.request("cancel_generation")
            except BackendError as error:
                self.store.log("WARN", "peer_cancel_failed", error.message)
        self.inference.cancel()
        self.server_engine.cancel()

    async def run_inference_benchmark(self, model_id: str, proxy_peer: bool = True) -> list[dict[str, Any]]:
        model = self._model(model_id)
        active_model_id = (
            self.server_engine.model_id if self.server_engine.active else self.inference.model_id
        )
        runs_on_peer = proxy_peer and active_model_id != model_id and (
            model.get("remoteOnly") or self.peer_active_model_id == model_id
        )
        if runs_on_peer:
            return await self.peer.request("benchmark_inference", {"modelId": model_id})
        load_started = time.perf_counter()
        temporary = active_model_id != model_id
        allocations: list[dict[str, Any]] = []
        error_message: str | None = None
        prompt = generation = 0.0
        try:
            if temporary:
                split = self.estimate_model_split(
                    model_id, {"contextSize": 4096, "gpuLayers": []}
                )
                allocations = [
                    {"nodeId": value["nodeId"], "layers": value["layers"], "kind": value["kind"]}
                    for value in split["devices"]
                ]
                await self.start_cluster(
                    model_id,
                    {"contextSize": 4096, "gpuLayers": allocations},
                    allow_peer=proxy_peer,
                    save_config=False,
                )
            if self.server_engine.active:
                prompt, generation = await self.server_engine.benchmark()
            else:
                prompt, generation = await self.inference.benchmark()
        except Exception as error:
            error_message = str(error)
        finally:
            load_time = time.perf_counter() - load_started if temporary else 0.0
            result = {
                "id": str(uuid.uuid4()), "modelName": model["name"],
                "topology": "distributed" if self.cluster.get("workerNodeId") else "local",
                "gpuLayers": allocations, "promptTokensPerSecond": prompt,
                "generationTokensPerSecond": generation, "loadTimeSeconds": load_time,
                "memoryPeakGb": 0.0, "recommended": error_message is None,
                "ranAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if error_message:
                result["error"] = error_message
            self.store.append_benchmark(result)
            if temporary:
                await self.stop_cluster(proxy_peer=False)
        return [result]

    def get_api_config(self) -> dict[str, Any]:
        port = int(self.store.get("apiPort", 11435))
        healthy = self.api_health() if self.api_health else False
        return {
            "url": f"http://127.0.0.1:{port}",
            "apiKey": self.store.get("apiKey"),
            "authRequired": bool(self.store.get("authRequired", True)),
            "healthy": healthy,
        }

    def regenerate_api_key(self) -> dict[str, Any]:
        self.store.update(apiKey=secrets.token_urlsafe(32))
        return self.get_api_config()

    async def try_api_request(self) -> dict[str, Any]:
        """Run the API page's example chat completion against the real loopback server."""
        port = int(self.store.get("apiPort", 11435))
        headers = {"Content-Type": "application/json"}
        if bool(self.store.get("authRequired", True)):
            headers["Authorization"] = f"Bearer {self.store.get('apiKey')}"
        payload = {
            "model": "active",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64,
        }
        started = time.perf_counter()
        status, body = await _loopback_post(port, "/v1/chat/completions", headers, payload)
        return {
            "status": status,
            "durationMs": round((time.perf_counter() - started) * 1000),
            "body": body[:_MAX_TRY_BODY_CHARS],
        }

    def _model(self, model_id: str) -> dict[str, Any]:
        model = next((value for value in self.models if value["id"] == model_id), None)
        if not model:
            raise BackendError("model_not_found", "Refresh the catalogue and choose an available model.")
        return model


def _port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


_TRY_TIMEOUT_SECONDS = 120
_MAX_TRY_BODY_CHARS = 20000


async def _loopback_post(
    port: int, path: str, headers: dict[str, str], payload: dict[str, Any]
) -> tuple[int, str]:
    """POST JSON to the local API server and return (HTTP status, body text)."""
    body = json.dumps(payload).encode()
    request = "\r\n".join(
        [
            f"POST {path} HTTP/1.1",
            f"Host: 127.0.0.1:{port}",
            "Connection: close",
            *[f"{key}: {value}" for key, value in headers.items()],
            f"Content-Length: {len(body)}",
        ]
    )
    writer: asyncio.StreamWriter | None = None
    raw = bytearray()
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 5)
        writer.write(request.encode() + b"\r\n\r\n" + body)
        await writer.drain()
        while True:
            chunk = await asyncio.wait_for(reader.read(64 * 1024), _TRY_TIMEOUT_SECONDS)
            if not chunk:
                break
            raw.extend(chunk)
    except (OSError, asyncio.TimeoutError) as error:
        raise BackendError(
            "api_unavailable",
            f"The local API did not answer on 127.0.0.1:{port}.",
            "Start the cluster or check the API port in Settings.",
        ) from error
    finally:
        if writer is not None:
            writer.close()
    text = bytes(raw).decode("utf-8", errors="replace")
    header_block, _, response_body = text.partition("\r\n\r\n")
    status_line = header_block.splitlines()[0] if header_block else ""
    parts = status_line.split()
    status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return status, response_body
