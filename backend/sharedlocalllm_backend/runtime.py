from __future__ import annotations

import inspect
import secrets
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from .errors import BackendError
from .hardware import probe_node
from .inference import InferenceEngine
from .models import discover_local, lm_studio_roots, merge_remote
from .peer import PeerManager
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
        self.inference = InferenceEngine(self.store)
        self.peer = PeerManager(self)
        self.api_port_changed: Callable[[int], Awaitable[None]] | None = None
        self.discover_models()

    async def start(self) -> None:
        await self.peer.start()
        await self.refresh_peer()

    async def stop(self) -> None:
        await self.inference.unload()
        await self.peer.stop()

    async def refresh_peer(self) -> None:
        peer = self.store.get("peer")
        if not peer:
            self._merge_models()
            return
        node = await self.peer.heartbeat()
        if node:
            node["online"] = True
            node["role"] = "worker"
        elif peer.get("capabilities"):
            peer["capabilities"]["online"] = False
            self.store.update(peer=peer)
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

    def snapshot(self) -> dict[str, Any]:
        nodes = [self.local_node]
        peer = self._peer_node()
        if peer:
            nodes.append(peer)
        return {
            "setupComplete": bool(self.store.get("setupComplete")),
            "runtime": {"status": "ready", "version": self._llama_version()},
            "deviceName": self.local_node["name"],
            "apiPort": int(self.store.get("apiPort", 11435)),
            "autostart": bool(self.store.get("autostart")),
            "nodes": nodes,
            "models": self.models,
            "modelDirectories": self._directories(),
            "network": self.network,
            "cluster": self.cluster,
            "benchmarks": self.store.get("benchmarks", []),
            "logs": self.store.logs(),
        }

    def _llama_version(self) -> str:
        try:
            import llama_cpp
            return f"llama-cpp-python {llama_cpp.__version__}"
        except Exception as error:
            return f"llama-cpp-python unavailable: {error}"

    async def dispatch(self, command: str, args: dict[str, Any]) -> Any:
        handlers = {
            "get_app_snapshot": lambda: self.snapshot(),
            "complete_setup": lambda: self.complete_setup(args["deviceName"]),
            "update_settings": lambda: self.update_settings(args["settings"]),
            "install_runtime": lambda: self.snapshot(),
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
        }
        handler = handlers.get(command)
        if not handler:
            raise BackendError("command_unknown", f"Unknown backend command: {command}")
        value = handler()
        if inspect.isawaitable(value):
            return await value
        return value

    def complete_setup(self, device_name: str) -> dict[str, Any]:
        name = device_name.strip()
        if not name or len(name) > 80:
            raise BackendError("device_name_invalid", "Device name must contain between 1 and 80 characters.")
        self.local_node["name"] = name
        self.store.update(deviceName=name, setupComplete=True)
        return self.snapshot()

    async def update_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        name = str(settings.get("deviceName", "")).strip()
        port = int(settings.get("apiPort", 11435))
        if not name or len(name) > 80 or not 1024 <= port <= 65535 or port == 11436:
            raise BackendError("settings_invalid", "Use a valid device name and API port from 1024-65535.")
        old_port = int(self.store.get("apiPort", 11435))
        if port != old_port and not _port_available(port):
            raise BackendError(
                "api_port_in_use", f"127.0.0.1:{port} is already in use.", "Choose another local API port."
            )
        self.local_node["name"] = name
        self.store.update(deviceName=name, apiPort=port, autostart=bool(settings.get("autostart")))
        if port != old_port and self.api_port_changed:
            await self.api_port_changed(port)
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
        self._merge_models()
        return node

    async def reset_pairing(self) -> dict[str, Any]:
        self.store.update(peer=None)
        self.peer.remote_models = []
        self._merge_models()
        return self.snapshot()

    async def run_network_test(self) -> dict[str, Any]:
        self.network = await self.peer.network_benchmark()
        return self.network

    def estimate_model_split(self, model_id: str, config: dict[str, Any]) -> dict[str, Any]:
        model = self._model(model_id)
        total = max(1, int(model.get("layerCount") or 1))
        allocations = list(config.get("gpuLayers") or [])
        if not allocations:
            nodes = [node for node in self.snapshot()["nodes"] if node.get("online", True)]
            total_vram = sum(max(0.1, float(node["gpu"].get("vramAvailableGb", 0))) for node in nodes)
            remaining = total
            for index, node in enumerate(nodes):
                layers = remaining if index == len(nodes) - 1 else round(
                    total * float(node["gpu"].get("vramAvailableGb", 0)) / total_vram
                )
                layers = max(0, min(remaining, layers))
                allocations.append({"nodeId": node["id"], "layers": layers})
                remaining -= layers
        per_layer = model["sizeBytes"] / total
        node_map = {node["id"]: node for node in self.snapshot()["nodes"]}
        devices = []
        gpu_layers = 0
        for allocation in allocations:
            layers = int(allocation.get("layers", 0))
            gpu_layers += layers
            node = node_map.get(allocation["nodeId"], {"gpu": {"vramAvailableGb": 0}})
            estimated = int(per_layer * layers / 1024**2 + 384)
            available = int(float(node["gpu"].get("vramAvailableGb", 0)) * 1024)
            devices.append({
                "nodeId": allocation["nodeId"], "layers": layers, "estimatedVramMib": estimated,
                "availableVramMib": available, "fits": estimated <= available,
            })
        return {
            "totalLayers": total, "gpuLayers": min(total, gpu_layers), "cpuLayers": max(0, total - gpu_layers),
            "estimatedCpuRamMib": int(per_layer * max(0, total - gpu_layers) / 1024**2),
            "usesAttentionMetadata": bool(model.get("attentionHeadCountKv")), "devices": devices,
        }

    async def start_cluster(
        self, model_id: str, load_config: dict[str, Any], allow_peer: bool = True
    ) -> dict[str, Any]:
        model = self._model(model_id)
        peer = self.store.get("peer")
        if model.get("remoteOnly"):
            if not allow_peer or not peer:
                raise BackendError("model_not_local", "This model is stored only on the other computer.")
            await self.peer.request("start_cluster", {"modelId": model_id, "loadConfig": load_config})
            self.cluster = {
                "status": "running", "coordinatorNodeId": peer["id"],
                "workerNodeId": self.local_node["id"], "modelId": model_id,
            }
            self._publish_local_cluster()
            return self.cluster
        path = self.model_paths.get(model_id)
        if not path:
            raise BackendError("model_not_found", "The selected local GGUF is unavailable.")
        peer_id = peer["id"] if peer else None
        self.cluster = {"status": "loading", "coordinatorNodeId": self.local_node["id"], "modelId": model_id}
        self._publish_local_cluster()
        await self.inference.load(model, path, load_config, self.peer, self.local_node["id"], peer_id)
        uses_peer = bool(peer_id and any(
            x.get("nodeId") == peer_id and x.get("layers", 0) > 0
            for x in load_config.get("gpuLayers", [])
        ))
        self.cluster = {
            "status": "running", "coordinatorNodeId": self.local_node["id"],
            "workerNodeId": peer_id if uses_peer else None, "modelId": model_id,
        }
        self._publish_local_cluster()
        return self.cluster

    async def stop_cluster(self, proxy_peer: bool = True) -> dict[str, Any]:
        coordinator = self.cluster.get("coordinatorNodeId")
        if proxy_peer and coordinator and coordinator != self.local_node["id"] and self.store.get("peer"):
            try:
                await self.peer.request("stop_cluster")
            except BackendError:
                pass
        await self.inference.unload()
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
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str], proxy_peer: bool = True
    ) -> dict[str, str]:
        coordinator = self.cluster.get("coordinatorNodeId")
        if proxy_peer and coordinator and coordinator != self.local_node["id"]:
            return await self.peer.request("chat", {"messages": messages, "settings": settings, "images": images})
        content = await self.inference.chat(messages, settings, images)
        return {"content": content}

    def cancel_generation(self) -> None:
        self.inference.cancel()

    async def run_inference_benchmark(self, model_id: str, proxy_peer: bool = True) -> list[dict[str, Any]]:
        model = self._model(model_id)
        if model.get("remoteOnly") and proxy_peer:
            return await self.peer.request("benchmark_inference", {"modelId": model_id})
        load_started = time.perf_counter()
        temporary = self.inference.model_id != model_id
        if temporary:
            split = self.estimate_model_split(model_id, {"contextSize": 4096, "gpuLayers": []})
            allocations = [{"nodeId": value["nodeId"], "layers": value["layers"]} for value in split["devices"]]
            await self.start_cluster(
                model_id, {"contextSize": 4096, "gpuLayers": allocations}, allow_peer=proxy_peer
            )
        load_time = time.perf_counter() - load_started if temporary else 0.0
        prompt, generation = await self.inference.benchmark()
        result = {
            "id": str(uuid.uuid4()), "modelName": model["name"],
            "topology": "distributed" if self.cluster.get("workerNodeId") else "local",
            "gpuLayers": [], "promptTokensPerSecond": prompt, "generationTokensPerSecond": generation,
            "loadTimeSeconds": load_time, "memoryPeakGb": 0.0,
            "recommended": True, "ranAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.store.append_benchmark(result)
        if temporary:
            await self.stop_cluster(proxy_peer=False)
        return [result]

    def get_api_config(self) -> dict[str, Any]:
        port = int(self.store.get("apiPort", 11435))
        return {"url": f"http://127.0.0.1:{port}", "apiKey": self.store.get("apiKey"), "healthy": True}

    def regenerate_api_key(self) -> dict[str, Any]:
        self.store.update(apiKey=secrets.token_urlsafe(32))
        return self.get_api_config()

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
