from __future__ import annotations

import math
from typing import Any

from .errors import BackendError

MIB = 1_048_576
GPU_RUNTIME_ALLOWANCE_MIB = 512
MIN_CONTEXT = 512
MAX_BATCH_SIZE = 4096


def active_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [node for node in nodes if node.get("online", True)]


def distribute_layers(total_layers: int, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    capable = [
        node
        for node in active_nodes(nodes)
        if float(node.get("gpu", {}).get("vramAvailableGb", 0)) > 0
    ]
    if total_layers <= 0 or not capable:
        return []
    total_vram = sum(float(node["gpu"]["vramAvailableGb"]) for node in capable)
    assigned = 0
    allocations: list[dict[str, Any]] = []
    for index, node in enumerate(capable):
        if index == len(capable) - 1:
            layers = total_layers - assigned
        else:
            layers = math.floor(
                total_layers * float(node["gpu"]["vramAvailableGb"]) / total_vram
            )
        assigned += layers
        allocations.append({"nodeId": node["id"], "layers": layers, "kind": "gpu"})
    return allocations


def normalize_load_config(
    model: dict[str, Any], config: dict[str, Any], nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    normalized = dict(config)
    model_context = max(MIN_CONTEXT, int(model.get("contextLength") or 4096))
    try:
        requested_context = int(config.get("contextSize") or 4096)
    except (TypeError, ValueError) as error:
        raise BackendError("context_invalid", "Context size must be a whole number.") from error
    normalized["contextSize"] = max(MIN_CONTEXT, min(requested_context, model_context))

    # Batch size comes from peer-forwarded load configs; a huge value makes
    # llama.cpp allocate proportional compute buffers and can OOM the machine.
    try:
        requested_batch = int(config.get("batchSize") or 512)
    except (TypeError, ValueError) as error:
        raise BackendError(
            "batch_size_invalid", "Batch size must be a whole number."
        ) from error
    normalized["batchSize"] = max(1, min(requested_batch, MAX_BATCH_SIZE))

    total_layers = int(model.get("layerCount") or 0)
    raw_allocations = config.get("gpuLayers") or []
    automatic = not raw_allocations
    normalized["automaticGpuOffload"] = total_layers <= 0 and not raw_allocations
    if not isinstance(raw_allocations, list):
        raise invalid_split("GPU layer allocations must be a list.")
    if automatic and total_layers:
        for gpu_layer_count in range(total_layers, 0, -1):
            candidate = distribute_layers(gpu_layer_count, nodes)
            candidate_config = {**normalized, "gpuLayers": candidate}
            candidate_estimate = estimate_split(model, candidate_config, nodes)
            if (
                candidate
                and candidate_estimate["cpuFits"]
                and all(device["fits"] for device in candidate_estimate["devices"])
            ):
                raw_allocations = candidate
                break

    node_map = {node["id"]: node for node in active_nodes(nodes)}
    seen: set[tuple[str, str]] = set()
    allocations: list[dict[str, Any]] = []
    assigned = 0
    for value in raw_allocations:
        if not isinstance(value, dict):
            raise invalid_split("Every layer allocation must be an object.")
        node_id = str(value.get("nodeId") or "")
        kind = str(value.get("kind") or "gpu")
        if node_id not in node_map:
            raise invalid_split("The split includes a computer that is not online in this cluster.")
        if kind not in ("gpu", "cpu"):
            raise invalid_split("Layer allocations must target either a GPU or CPU.")
        if kind == "cpu" and not bool(config.get("includeRemoteCpu")):
            raise invalid_split("Remote CPU layers require the remote CPU option to be enabled.")
        if kind == "cpu" and nodes and node_id == nodes[0].get("id"):
            raise invalid_split("Local CPU layers are implicit and must not be allocated explicitly.")
        try:
            raw_layers = value.get("layers", 0)
            if isinstance(raw_layers, bool):
                raise ValueError
            layers = int(raw_layers)
            if isinstance(raw_layers, float) and raw_layers != layers:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise invalid_split("Layer counts must be whole numbers.") from error
        if layers < 0:
            raise invalid_split("Layer counts cannot be negative.")
        key = (node_id, kind)
        if key in seen:
            raise invalid_split("A computer target appears more than once in the split.")
        seen.add(key)
        assigned += layers
        allocations.append({"nodeId": node_id, "layers": layers, "kind": kind})

    if total_layers and assigned > total_layers:
        raise invalid_split(
            f"The split assigns {assigned} layers, but the model has only {total_layers}."
        )
    if raw_allocations and total_layers and assigned == 0:
        raise invalid_split("Assign at least one layer, or use automatic allocation.")
    normalized["gpuLayers"] = allocations
    return normalized


def estimate_split(
    model: dict[str, Any], config: dict[str, Any], nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    total_layers = int(model.get("layerCount") or 0)
    if total_layers <= 0:
        raise BackendError(
            "layer_metadata_missing",
            "This GGUF does not report a usable layer count.",
            "Use automatic allocation or refresh the catalogue.",
        )
    allocations = list(config.get("gpuLayers") or [])
    node_map = {node["id"]: node for node in active_nodes(nodes)}
    model_mib = math.ceil(int(model["sizeBytes"]) / MIB)
    context = int(config["contextSize"])
    embedding = int(model.get("embeddingLength") or 0)
    heads = int(model.get("attentionHeadCount") or 0)
    kv_heads = int(model.get("attentionHeadCountKv") or 0)
    has_attention = embedding > 0 and heads > 0 and kv_heads > 0
    kv_mib_per_layer = (
        math.ceil(context * math.ceil(embedding / heads) * kv_heads * 2 * 2 / MIB)
        if has_attention
        else math.ceil(context / 4096) * 16
    )

    devices: list[dict[str, Any]] = []
    gpu_layers = 0
    assigned_layers = 0
    for allocation in allocations:
        node = node_map[allocation["nodeId"]]
        layers = int(allocation["layers"])
        assigned_layers += layers
        weight_mib = math.ceil(model_mib * layers / total_layers)
        if allocation.get("kind") == "cpu":
            available = math.floor(max(0.0, float(node.get("ramAvailableGb", 0))) * 1024)
            estimated = weight_mib
            kind = "cpu"
        else:
            gpu_layers += layers
            available = math.floor(
                max(0.0, float(node.get("gpu", {}).get("vramAvailableGb", 0))) * 1024
            )
            estimated = (
                weight_mib + kv_mib_per_layer * layers + GPU_RUNTIME_ALLOWANCE_MIB
                if layers
                else 0
            )
            kind = "gpu"
        devices.append(
            {
                "nodeId": allocation["nodeId"],
                "layers": layers,
                "kind": kind,
                "estimatedVramMib": estimated,
                "availableVramMib": available,
                "fits": estimated <= available,
            }
        )

    cpu_layers = max(0, total_layers - assigned_layers)
    estimated_cpu_ram = math.ceil(model_mib * cpu_layers / total_layers)
    available_cpu_ram = math.floor(
        max(0.0, float(nodes[0].get("ramAvailableGb", 0))) * 1024
    ) if nodes else 0
    return {
        "totalLayers": total_layers,
        "gpuLayers": gpu_layers,
        "cpuLayers": cpu_layers,
        "estimatedCpuRamMib": estimated_cpu_ram,
        "availableCpuRamMib": available_cpu_ram,
        "cpuFits": estimated_cpu_ram <= available_cpu_ram,
        "usesAttentionMetadata": has_attention,
        "devices": devices,
    }


def validate_fit(estimate: dict[str, Any], force: bool) -> None:
    if force:
        return
    failed = next((device for device in estimate["devices"] if not device["fits"]), None)
    if failed:
        raise BackendError(
            "split_exceeds_capacity",
            f"The selected device needs about {failed['estimatedVramMib']} MiB, "
            f"but only {failed['availableVramMib']} MiB is available.",
            "Move layers, reduce context size, or enable force launch.",
        )
    if not estimate.get("cpuFits", True):
        raise BackendError(
            "split_exceeds_capacity",
            f"Local CPU layers need about {estimate['estimatedCpuRamMib']} MiB of RAM, "
            f"but only {estimate['availableCpuRamMib']} MiB is available.",
            "Move more layers to a GPU, reduce model/context size, or enable force launch.",
        )


def invalid_split(message: str) -> BackendError:
    return BackendError("invalid_layer_split", message, "Adjust the per-device layer counts.")
