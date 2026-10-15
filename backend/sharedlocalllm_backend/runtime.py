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

from .autotune import Autotuner, apply_tune_result, locate_bench, topology_fingerprint
from .errors import BackendError
from .hardware import probe_node
from .inference import InferenceEngine, layer_totals_multi
from .llama_server import install_root_candidates
from .load_progress import LoadStage, cluster_workers, loading_cluster
from .models import discover_local, hf_cache_roots, lm_studio_roots, merge_remote_multi, refresh_fits
from .peer import PeerManager, RpcForwarder
from .placement import estimate_split, expected_transfer_bytes, normalize_load_config, validate_fit
from .rpc_native import runtime_health
from .server_engine import START_TIMEOUT_SECONDS, ServerEngine
from .store import Store, save_peers, stored_peers


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
        # Model each paired computer is currently running, by peer id. Used to
        # route proxied chat/benchmarks to the right computer in 3-node clusters.
        self.peer_active_models: dict[str, str] = {}
        self.inference = InferenceEngine(self.store)
        self.peer = PeerManager(self)
        self.server_engine = ServerEngine(self.store)
        self.autotuner = Autotuner(self.store)
        self._active_compute_operation: str | None = None
        self._server_forwarders: list[RpcForwarder] = []
        self.api_port_changed: Callable[[int], Awaitable[None]] | None = None
        self.api_health: Callable[[], bool] | None = None
        self._peer_refresh_task: asyncio.Task[None] | None = None
        self._runtime = runtime_health()
        self.discover_models()

    @property
    def _server_forwarder(self) -> RpcForwarder | None:
        """First llama-server RPC forwarder (legacy single-worker readers)."""
        return self._server_forwarders[0] if self._server_forwarders else None

    @_server_forwarder.setter
    def _server_forwarder(self, value: RpcForwarder | None) -> None:
        self._server_forwarders = [value] if value is not None else []

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
        await self.autotuner.cancel()
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
        peers = stored_peers(self.store)
        if not peers:
            self.peer_active_models = {}
            self._merge_models()
            return
        for peer in peers:
            peer_id = str(peer.get("id"))
            node = await self.peer.heartbeat_peer(peer_id)
            if node:
                node["online"] = True
                node["role"] = "worker"
                if node.get("clusterStatus") == "running" and node.get("clusterModelId"):
                    self.peer_active_models[peer_id] = str(node["clusterModelId"])
                else:
                    self.peer_active_models.pop(peer_id, None)
            else:
                self.peer_active_models.pop(peer_id, None)
        refresh_fits(self.local_models, self.local_node, self._peer_nodes())
        self._merge_models()

    def discover_models(self) -> list[dict[str, Any]]:
        roots = lm_studio_roots() + hf_cache_roots() + [Path(value) for value in self.store.get("customModelDirectories", [])]
        self.local_models, self.model_paths = discover_local(
            roots, self.local_node["id"], self.local_node, self._peer_nodes()
        )
        self._merge_models()
        return self.models

    def _merge_models(self) -> None:
        models_by_peer: dict[str, list[dict[str, Any]]] = {}
        for peer in stored_peers(self.store):
            peer_id = str(peer.get("id"))
            models_by_peer[peer_id] = self.peer.remote_models_by_peer.get(peer_id, [])
        self.models = merge_remote_multi(
            self.local_models, models_by_peer, self.local_node["id"]
        )

    def _peer_nodes(self) -> list[dict[str, Any]]:
        """Capabilities of every paired computer (online or last-known)."""
        nodes: list[dict[str, Any]] = []
        for peer in stored_peers(self.store):
            capabilities = peer.get("capabilities")
            if isinstance(capabilities, dict):
                nodes.append(capabilities)
            else:
                nodes.append({
                    "id": peer.get("id"), "name": peer.get("name", "Paired computer"),
                    "online": False, "role": "worker",
                    "gpu": {"vramAvailableGb": 0.0}, "ramAvailableGb": 0.0,
                })
        return nodes

    def _peer_node(self) -> dict[str, Any] | None:
        """First paired computer (legacy single-peer callers)."""
        nodes = self._peer_nodes()
        return nodes[0] if nodes else None

    def _cluster_nodes(self) -> list[dict[str, Any]]:
        nodes = [self.local_node]
        nodes.extend(peer for peer in self._peer_nodes() if peer.get("online", True))
        return nodes

    def _begin_compute_operation(self, operation: str) -> None:
        if self._active_compute_operation:
            raise BackendError(
                "compute_busy",
                f"Cannot start {operation} while {self._active_compute_operation} is in progress.",
                "Wait for the current model operation to finish or cancel it first.",
            )
        self._active_compute_operation = operation

    def _end_compute_operation(self, operation: str) -> None:
        if self._active_compute_operation == operation:
            self._active_compute_operation = None

    def snapshot(self) -> dict[str, Any]:
        nodes = [self.local_node, *self._peer_nodes()]
        return {
            "setupComplete": bool(self.store.get("setupComplete")),
            "runtime": dict(self._runtime),
            "deviceName": self.local_node["name"],
            "deviceId": self.local_node["id"],
            "apiPort": int(self.store.get("apiPort", 11435)),
            "authRequired": bool(self.store.get("authRequired", True)),
            "autostart": bool(self.store.get("autostart")),
            "nodes": nodes,
            "models": self.models,
            "modelLoadConfigs": self.store.model_load_configs(),
            "modelTunes": self.store.model_tunes(),
            "modelDirectories": self._directories(),
            "network": self.network,
            "cluster": self._cluster_snapshot(),
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
            "forget_peer": lambda: self.forget_peer(str(args.get("peerId") or "")),
            "estimate_model_split": lambda: self.estimate_model_split(args["modelId"], args["loadConfig"]),
            "start_cluster": lambda: self.start_cluster(args["modelId"], args["loadConfig"]),
            "stop_cluster": self.stop_cluster,
            "run_inference_benchmark": lambda: self.run_inference_benchmark(args["modelId"]),
            "cancel_inference_benchmark": self.cancel_generation,
            "start_model_autotune": lambda: self.start_model_autotune(
                args["modelId"], args.get("depth", "quick")
            ),
            "get_autotune_status": self.autotuner.status,
            "cancel_model_autotune": self.autotuner.cancel,
            "apply_model_tune": lambda: self.apply_model_tune(args["modelId"]),
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
                "Run bun backend:install in development, or reinstall the desktop application.",
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
        refresh_fits(self.local_models, self.local_node, self._peer_nodes())
        self._merge_models()
        return node

    async def forget_peer(self, peer_id: str) -> dict[str, Any]:
        """Forget one paired computer without dropping the others (3-node clusters)."""
        if not peer_id or not _find_peer(stored_peers(self.store), peer_id):
            raise BackendError("peer_unknown", "That computer is not paired.")
        coordinator_id = self.cluster.get("coordinatorNodeId")
        if peer_id in cluster_workers(self.cluster) or (
            coordinator_id == peer_id and coordinator_id != self.local_node["id"]
        ):
            try:
                await self.stop_cluster(proxy_peer=False)
            except BackendError:
                await self.inference.unload()
                await self.server_engine.stop()
                await self._stop_server_forwarder()
                self.cluster = {"status": "idle"}
                self._publish_local_cluster()
        peers = [peer for peer in stored_peers(self.store) if str(peer.get("id")) != peer_id]
        save_peers(self.store, peers)
        self.peer.forget_peer_models(peer_id)
        self.peer_active_models.pop(peer_id, None)
        refresh_fits(self.local_models, self.local_node, self._peer_nodes())
        self._merge_models()
        self.store.log("INFO", "peer_forgotten", "Forgot one paired computer")
        return self.snapshot()

    async def reset_pairing(self) -> dict[str, Any]:
        if self.cluster.get("status") in ("loading", "running", "error"):
            try:
                await self.stop_cluster()
            except BackendError:
                await self.inference.unload()
                await self.server_engine.stop()
                await self._stop_server_forwarder()
        save_peers(self.store, [])
        self.peer.remote_models = []
        self.peer.remote_models_by_peer = {}
        self.peer_active_models = {}
        self.network = None
        self.cluster = {"status": "idle"}
        self._publish_local_cluster()
        refresh_fits(self.local_models, self.local_node, [])
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
        operation = "cluster startup"
        self._begin_compute_operation(operation)
        try:
            return await self._start_cluster(model_id, load_config, allow_peer, save_config)
        finally:
            self._end_compute_operation(operation)

    async def _start_cluster(
        self, model_id: str, load_config: dict[str, Any], allow_peer: bool = True,
        save_config: bool = True,
    ) -> dict[str, Any]:
        model = self._model(model_id)
        peers = {str(peer.get("id")): peer for peer in stored_peers(self.store)}
        if model.get("remoteOnly"):
            if not allow_peer or not peers:
                raise BackendError("model_not_local", "This model is stored only on another computer.")
            coordinator_id = next(
                (node_id for location in model.get("locations", [])
                 if (node_id := str(location.get("nodeId"))) in peers),
                next(iter(peers)),
            )
            await self.peer.request_to_peer(
                coordinator_id, "start_cluster", {"modelId": model_id, "loadConfig": load_config}
            )
            if save_config:
                self.store.save_model_load_config(model_id, load_config)
            self.cluster = {
                "status": "running", "coordinatorNodeId": coordinator_id,
                "workerNodeId": self.local_node["id"], "workerNodeIds": [self.local_node["id"]],
                "modelId": model_id,
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
        include_remote_cpu = bool(normalized_config.get("includeRemoteCpu"))
        allocations = normalized_config.get("gpuLayers") or []
        candidate_ids = sorted({
            str(allocation.get("nodeId"))
            for allocation in allocations
            if str(allocation.get("nodeId")) in peers
            and int(allocation.get("layers", 0)) > 0
        })
        if model.get("fit") == "combined-gpu" and not candidate_ids:
            candidate_ids = sorted(
                peer_id for peer_id, peer in peers.items()
                if (peer.get("capabilities") or {}).get("online", True)
            )
        worker_ids: list[str] = []
        for worker_id in candidate_ids:
            if await self.peer.heartbeat_peer(worker_id):
                worker_ids.append(worker_id)
            elif model.get("fit") == "combined-gpu":
                raise BackendError(
                    "peer_unavailable",
                    "A paired computer did not answer, and this model needs all GPUs.",
                    "Bring every worker online or choose a model that fits locally.",
                )
            else:
                self.store.log(
                    "WARN", "peer_offline_launch_skipped",
                    f"Launching without unreachable worker {worker_id}",
                )
        if candidate_ids and not worker_ids and model.get("fit") != "combined-gpu":
            normalized_config["gpuLayers"] = [
                allocation for allocation in allocations
                if str(allocation.get("nodeId")) not in candidate_ids
            ]
            allocations = normalized_config["gpuLayers"]
        per_worker, local_layers = layer_totals_multi(
            allocations, worker_ids, self.local_node["id"], include_remote_cpu,
        )
        remote_gpu = sum(values[0] for values in per_worker.values())
        remote_cpu = sum(values[1] for values in per_worker.values())
        engine = str(
            load_config.get("engine")
            or ("llama-server" if model.get("mtp") else "builtin")
        )
        no_load_timeout = bool(normalized_config.get("noLoadTimeout"))
        expected_bytes = self._expected_transfer_bytes(model, normalized_config, worker_ids)
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
                self.cluster = loading_cluster(
                    model_id, self.local_node["id"], worker_ids[0] if worker_ids else None,
                    "llama-server", expected_bytes, worker_ids=worker_ids or None,
                )
                self._publish_local_cluster()
                try:
                    rpc_endpoints: list[str] = []
                    if worker_ids:
                        self._set_load_stage(model_id, "tunnel")
                        # Publish ownership before starting, so polls see live
                        # counters and every preparation failure can clean up.
                        for worker_id in worker_ids:
                            gpu_layers, cpu_layers = per_worker.get(worker_id, (0, 0))
                            forwarder = RpcForwarder(
                                self.peer, model_id=model_id, include_cpu=cpu_layers > 0,
                                peer_id=worker_id,
                            )
                            # Track before starting so a failed start is still
                            # cleaned up by _stop_server_forwarder.
                            self._server_forwarders.append(forwarder)
                            rpc_endpoints.append(await forwarder.start())
                    tensor_split: list[int] | None = None
                    gpu_layers: int | None = None
                    if allocations:
                        gpu_layers = remote_gpu + remote_cpu + local_layers
                        tensor_split = []
                        for worker_id in worker_ids:
                            gpu_layers_w, cpu_layers_w = per_worker.get(worker_id, (0, 0))
                            worker_node = (peers.get(worker_id) or {}).get("capabilities", {})
                            if rpc_endpoints and (gpu_layers_w > 0 or _node_has_gpu(worker_node)):
                                tensor_split.append(gpu_layers_w)
                            if rpc_endpoints and cpu_layers_w > 0:
                                tensor_split.append(cpu_layers_w)
                        if local_layers > 0 or _node_has_gpu(self.local_node):
                            tensor_split.append(local_layers)
                    self._set_load_stage(model_id, "loading_weights")
                    if no_load_timeout:
                        self.store.log(
                            "INFO", "llama_server_no_timeout",
                            "Startup timeout disabled for this launch; waiting as long as it takes.",
                        )
                    await self.server_engine.start(
                        exe=exe, model_path=path, model_id=model_id,
                        context=int(normalized_config.get("contextSize", 4096)),
                        api_key=self.store.get("apiKey"), mtp=bool(model.get("mtp")),
                        rpc_endpoint=rpc_endpoints[0] if rpc_endpoints else None,
                        rpc_endpoints=rpc_endpoints or None,
                        gpu_layers=gpu_layers,
                        tensor_split=tensor_split,
                        reasoning_preserve=bool(model.get("reasoningPreserve")),
                        load_config=normalized_config,
                        timeout_seconds=None if no_load_timeout else START_TIMEOUT_SECONDS,
                    )
                except (Exception, asyncio.CancelledError) as error:
                    await self.server_engine.stop()
                    await self._stop_server_forwarder()
                    self.cluster = {
                        "status": "error", "coordinatorNodeId": self.local_node["id"],
                        "modelId": model_id, "engine": "llama-server",
                        "error": str(error) or "llama-server startup was cancelled",
                    }
                    self._publish_local_cluster()
                    raise
                if save_config:
                    self.store.save_model_load_config(model_id, normalized_config)
                self.cluster = {
                    "status": "running", "coordinatorNodeId": self.local_node["id"],
                    "workerNodeId": worker_ids[0] if worker_ids else None,
                    "workerNodeIds": list(worker_ids) or None,
                    "modelId": model_id, "engine": "llama-server",
                }
                self._publish_local_cluster()
                return self.cluster
        await self.server_engine.stop()
        await self._stop_server_forwarder()
        self.cluster = loading_cluster(
            model_id, self.local_node["id"], worker_ids[0] if worker_ids else None,
            "builtin", expected_bytes, worker_ids=worker_ids or None,
        )
        self._publish_local_cluster()
        try:
            await self.inference.load(
                model, path, normalized_config, self.peer, self.local_node["id"], worker_ids[0] if worker_ids else None,
                on_stage=lambda stage: self._set_load_stage(model_id, stage),
                worker_ids=worker_ids,
            )
            if save_config:
                self.store.save_model_load_config(model_id, normalized_config)
            self.cluster = {
                "status": "running", "coordinatorNodeId": self.local_node["id"],
                "workerNodeId": worker_ids[0] if worker_ids else None,
                "workerNodeIds": list(worker_ids) or None,
                "modelId": model_id, "engine": "builtin",
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
        heartbeats: the first paired computer with a running model wins.
        """
        if self.cluster.get("status") in ("loading", "running", "stopping"):
            coordinator = self.cluster.get("coordinatorNodeId")
            if coordinator and coordinator != self.local_node["id"]:
                return str(coordinator)
            return None
        for peer in stored_peers(self.store):
            peer_id = str(peer.get("id"))
            if self.peer_active_models.get(peer_id):
                return peer_id
        return None

    async def _stop_server_forwarder(self) -> None:
        forwarders, self._server_forwarders = self._server_forwarders, []
        for forwarder in forwarders:
            await forwarder.stop()

    async def stop_cluster(self, proxy_peer: bool = True) -> dict[str, Any]:
        remote_coordinator = self._remote_coordinator() if proxy_peer else None
        if remote_coordinator and _find_peer(stored_peers(self.store), remote_coordinator):
            try:
                await self.peer.request_to_peer(remote_coordinator, "stop_cluster")
            except BackendError as error:
                self.cluster = {**self.cluster, "status": "error", "error": error.message}
                self._publish_local_cluster()
                raise
        await self.inference.unload()
        await self.server_engine.stop()
        await self._stop_server_forwarder()
        self.cluster = {"status": "ready" if stored_peers(self.store) else "idle"}
        self._publish_local_cluster()
        return self.cluster

    def _publish_local_cluster(self) -> None:
        self.local_node["clusterStatus"] = self.cluster.get("status")
        model_id = self.cluster.get("modelId")
        if model_id:
            self.local_node["clusterModelId"] = model_id
        else:
            self.local_node.pop("clusterModelId", None)

    def _set_load_stage(self, model_id: str, stage: LoadStage) -> None:
        """Advance the visible load stage while a cluster is starting.

        Late callbacks (for example after a stop) must not resurrect a
        finished load, so stages only apply to the matching model load.
        """
        if self.cluster.get("modelId") != model_id:
            return
        if self.cluster.get("status") != "loading":
            return
        self.cluster = {**self.cluster, "stage": stage}
        self._publish_local_cluster()

    def _expected_transfer_bytes(
        self, model: dict[str, Any], normalized_config: dict[str, Any], peer_ids: list[str] | None,
    ) -> int:
        allocations = normalized_config.get("gpuLayers") or []
        per_worker, _local = layer_totals_multi(
            allocations, list(peer_ids or []), self.local_node["id"],
            bool(normalized_config.get("includeRemoteCpu")),
        )
        remote_gpu = sum(values[0] for values in per_worker.values())
        remote_cpu = sum(values[1] for values in per_worker.values())
        total_layers = int(model.get("layerCount") or 0)
        return expected_transfer_bytes(
            int(model.get("sizeBytes") or 0), remote_gpu + remote_cpu, total_layers
        )

    def _cluster_snapshot(self) -> dict[str, Any]:
        """Snapshot the cluster with live load-progress counters attached.

        Forwarder byte counters are read here (on every snapshot poll) rather
        than cached, so the transfer meter moves while native code loads.
        With several workers the counters are summed across forwarders.
        """
        cluster = dict(self.cluster)
        if cluster.get("status") == "loading":
            forwarders = list(self._server_forwarders) + list(
                getattr(self.inference, "_forwarders", [])
            )
            if forwarders:
                cluster["bytesToWorker"] = sum(
                    int(getattr(forwarder, "bytes_to_worker", 0)) for forwarder in forwarders
                )
                cluster["bytesFromWorker"] = sum(
                    int(getattr(forwarder, "bytes_from_worker", 0)) for forwarder in forwarders
                )
        return cluster

    async def chat(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str],
        proxy_peer: bool = True, tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict[str, Any]:
        remote_coordinator = self._remote_coordinator() if proxy_peer else None
        if remote_coordinator:
            payload: dict[str, Any] = {
                "messages": messages, "settings": settings, "images": images,
                "tools": tools, "toolChoice": tool_choice,
            }
            return await self.peer.request_to_peer(remote_coordinator, "chat", payload)
        if self.server_engine.active:
            if images:
                raise BackendError(
                    "vision_not_migrated",
                    "Vision attachments are not enabled in the Python migration yet.",
                )
            return await self.server_engine.chat(messages, settings, tools, tool_choice)
        return await self.inference.chat(messages, settings, images, tools, tool_choice)

    async def chat_stream_events(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str],
        proxy_peer: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        remote_coordinator = self._remote_coordinator() if proxy_peer else None
        if remote_coordinator:
            async for event in self.peer.stream_to_peer(remote_coordinator, "chat_stream", {
                "messages": messages, "settings": settings, "images": images,
            }):
                yield event
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

    async def chat_openai_stream(
        self,
        messages: list[dict[str, Any]],
        settings: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        proxy_peer: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        remote_coordinator = self._remote_coordinator() if proxy_peer else None
        if remote_coordinator:
            async for event in self.peer.stream_to_peer(remote_coordinator, "chat_openai_stream", {
                "messages": messages, "settings": settings,
                "tools": tools, "toolChoice": tool_choice,
            }):
                yield event
            return
        engine = self.server_engine if self.server_engine.active else self.inference
        async for event in engine.chat_openai_stream(
            messages, settings, tools, tool_choice
        ):
            yield event

    async def cancel_generation(self) -> None:
        remote_coordinator = self._remote_coordinator()
        if remote_coordinator:
            try:
                await self.peer.request_to_peer(remote_coordinator, "cancel_generation")
            except BackendError as error:
                self.store.log("WARN", "peer_cancel_failed", error.message)
        self.inference.cancel()
        self.server_engine.cancel()

    def _peer_running_model(self, model: dict[str, Any]) -> str | None:
        """Id of the paired computer currently running this model, if any."""
        model_id = model.get("id")
        for peer in stored_peers(self.store):
            peer_id = str(peer.get("id"))
            if self.peer_active_models.get(peer_id) == model_id:
                return peer_id
        if model.get("remoteOnly"):
            locations = {
                str(location.get("nodeId")) for location in model.get("locations", [])
            }
            for peer in stored_peers(self.store):
                if str(peer.get("id")) in locations:
                    return str(peer.get("id"))
        return None

    async def run_inference_benchmark(self, model_id: str, proxy_peer: bool = True) -> list[dict[str, Any]]:
        operation = "inference benchmark"
        self._begin_compute_operation(operation)
        try:
            return await self._run_inference_benchmark(model_id, proxy_peer)
        finally:
            self._end_compute_operation(operation)

    async def _run_inference_benchmark(
        self, model_id: str, proxy_peer: bool = True,
    ) -> list[dict[str, Any]]:
        model = self._model(model_id)
        active_model_id = (
            self.server_engine.model_id if self.server_engine.active else self.inference.model_id
        )
        runs_on_peer = (
            proxy_peer
            and active_model_id != model_id
            and self._peer_running_model(model) is not None
        )
        if runs_on_peer:
            peer_id = self._peer_running_model(model)
            if peer_id is None:
                raise BackendError("peer_unavailable", "No paired computer runs that model.")
            return await self.peer.request_to_peer(peer_id, "benchmark_inference", {"modelId": model_id})
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
                await self._start_cluster(
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

    async def start_model_autotune(self, model_id: str, depth: str = "quick") -> dict[str, Any]:
        """Sweep llama-bench configurations for a local model.

        Requires an idle cluster (VRAM must be free) and the pinned
        llama-bench runtime; distributed tuning reuses the peer RPC tunnel.
        """
        operation = "model tuning"
        self._begin_compute_operation(operation)
        forwarder: RpcForwarder | None = None
        handed_off = False
        try:
            if self.cluster.get("status") in ("loading", "running"):
                raise BackendError(
                    "cluster_active",
                    "Stop the running cluster before tuning this model.",
                    "Benchmarks need the GPU memory that the loaded model is using.",
                )
            model = self._model(model_id)
            if model.get("remoteOnly"):
                raise BackendError(
                    "model_not_local",
                    "This model is stored only on the other computer.",
                    "Start tuning from the computer that stores the GGUF file.",
                )
            path = self.model_paths.get(model_id)
            if not path:
                raise BackendError("model_not_found", "The selected local GGUF is unavailable.")
            exe = locate_bench(install_root_candidates())
            if exe is None:
                raise BackendError(
                    "runtime_unavailable",
                    "The pinned llama-bench tool is not installed.",
                    "Run bun backend:install in development, or reinstall the desktop application.",
                )
            nodes = self._cluster_nodes()
            # Autotune measures the coordinator plus the first online worker;
            # full 3-node measurement is covered by split estimates.
            first_peer = next(
                (peer for peer in stored_peers(self.store)
                 if (peer.get("capabilities") or {}).get("online", True)),
                None,
            )
            rpc_endpoint = None
            if first_peer:
                forwarder = RpcForwarder(
                    self.peer, model_id=model_id,
                    peer_id=str(first_peer.get("id")),
                )
                rpc_endpoint = await forwarder.start()
            status = await self.autotuner.start(
                exe=exe, model_path=path, model_id=model_id,
                model_name=model["name"], depth=depth, nodes=nodes,
                total_layers=int(model.get("layerCount") or 0),
                rpc_endpoint=rpc_endpoint,
                fingerprint=topology_fingerprint(nodes),
                on_complete=lambda result: self._record_model_tune(model_id, result),
                on_finish=lambda _status: self._finish_model_autotune(forwarder),
            )
            handed_off = True
            return status
        finally:
            if not handed_off and forwarder:
                await forwarder.stop()
            if not handed_off:
                self._end_compute_operation(operation)

    async def _finish_model_autotune(self, forwarder: RpcForwarder | None) -> None:
        try:
            if forwarder:
                await forwarder.stop()
        finally:
            self._end_compute_operation("model tuning")

    def _record_model_tune(self, model_id: str, result: dict[str, Any]) -> None:
        winners = result.get("winners") or {}
        self.store.save_model_tune(model_id, result)
        self.store.append_benchmark({
            "id": str(uuid.uuid4()), "modelName": result.get("modelName", ""),
            "topology": "distributed" if winners.get("tensorSplit") else "local",
            "gpuLayers": winners.get("gpuLayersAllocations") or [],
            "promptTokensPerSecond": result.get("promptTokensPerSecond", 0.0),
            "generationTokensPerSecond": result.get("generationTokensPerSecond", 0.0),
            "loadTimeSeconds": 0.0, "memoryPeakGb": 0.0, "recommended": True,
            "autotune": True, "ranAt": result.get("ranAt", ""),
        })

    def apply_model_tune(self, model_id: str) -> dict[str, Any]:
        """Merge a stored tune into this model's load config for launch."""
        tune = self.store.model_tunes().get(model_id)
        if not tune:
            raise BackendError(
                "tune_missing",
                "This model has no saved tuning result.",
                "Run Auto-tune first.",
            )
        if tune.get("fingerprint") != topology_fingerprint(self._cluster_nodes()):
            raise BackendError(
                "tune_topology_changed",
                "The saved tune was measured on a different computer or GPU topology.",
                "Run Auto-tune again before applying these settings.",
            )
        base = dict(self.store.model_load_configs().get(model_id) or {})
        base.setdefault("contextSize", 4096)
        base.setdefault("gpuLayers", [])
        config = apply_tune_result(base, tune)
        # Persist immediately so the next launch prefills with tuned values
        # even if the user closes the app before starting a cluster.
        self.store.save_model_load_config(model_id, config)
        return {
            "loadConfig": config,
            "staleTopology": False,
            "tunedAt": tune.get("ranAt"),
        }

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


def _find_peer(peers: list[dict[str, Any]], peer_id: str) -> dict[str, Any] | None:
    return next((peer for peer in peers if str(peer.get("id")) == peer_id), None)


def _port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _node_has_gpu(node: dict[str, Any]) -> bool:
    gpu = node.get("gpu") or {}
    return max(
        float(gpu.get("vramTotalGb") or 0),
        float(gpu.get("vramAvailableGb") or 0),
    ) > 0


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
