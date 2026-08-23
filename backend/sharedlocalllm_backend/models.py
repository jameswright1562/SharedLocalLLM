from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .gguf import has_nextn_tensors, read_metadata

SHARD = re.compile(r"^(.*?)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
QUANT = re.compile(r"(?:^|[-_.])(Q\d(?:_[A-Z0-9]+)+|Q\d_[A-Z0-9]+)(?:[-_.]|$)", re.IGNORECASE)

# Catalogue fit must reflect the default launch, where the UI caps context at
# this size. Reserving KV cache for a model's native maximum (262K tokens on
# current Qwen releases) would brand every long-context GGUF does-not-fit even
# though a normal 8K-context load fits comfortably.
FIT_CONTEXT_TOKENS = 8192


def model_slug(name: str, quantization: str) -> str:
    """Stable lowercase alias for a catalogue entry, e.g. orchid-9b-q4_k_m."""
    base = name.strip().lower()
    quant = (quantization or "").strip().lower()
    label = base if not quant or quant in base else f"{base}-{quant}"
    return re.sub(r"\s+", "-", label)


def lm_studio_roots() -> list[Path]:
    home = Path.home()
    base = home / ".lmstudio"
    roots = [base / "models", base / "hub" / "models", base / ".internal" / "bundled-models"]
    try:
        value = json.loads((base / "settings.json").read_text(encoding="utf-8"))
        configured = value.get("downloadsFolder")
        if configured:
            path = Path(configured)
            roots.append(path if path.is_absolute() else base / path)
    except (OSError, json.JSONDecodeError):
        pass
    return [path for path in roots if path.is_dir()]


def _model_id(shards: list[Path], size: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    for path in shards:
        try:
            with path.open("rb") as handle:
                digest.update(handle.read(64 * 1024))
                if path.stat().st_size > 64 * 1024:
                    handle.seek(-64 * 1024, 2)
                    digest.update(handle.read(64 * 1024))
        except OSError:
            digest.update(path.name.lower().encode())
    return digest.hexdigest()[:24]


def _quantization(name: str) -> str:
    match = QUANT.search(name.upper())
    return match.group(1).upper() if match else "unknown"


def _fit(
    size: int,
    node: dict[str, Any],
    peer: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> str:
    # Include a conservative runtime/KV reserve instead of treating model bytes
    # alone as the complete GPU requirement. The reserve is sized for the
    # default launch context, not the native maximum (see FIT_CONTEXT_TOKENS).
    layers = max(1, int(metadata.get("layerCount") or 1))
    context = min(max(512, int(metadata.get("contextLength") or 4096)), FIT_CONTEXT_TOKENS)
    kv_fallback = layers * max(1, math.ceil(context / 4096)) * 16 * 1024**2
    required = size + kv_fallback + 512 * 1024**2
    local_vram = float(node.get("gpu", {}).get("vramAvailableGb", 0)) * 1024**3
    local_ram = float(node.get("ramAvailableGb", 0)) * 1024**3
    if required <= local_vram:
        return "single-node"
    peer_vram = 0.0
    if peer and peer.get("online", False):
        peer_vram = float(peer.get("gpu", {}).get("vramAvailableGb", 0)) * 1024**3
    if peer_vram and required + 512 * 1024**2 <= local_vram + peer_vram:
        return "combined-gpu"
    if required <= local_vram + local_ram:
        return "gpu-ram"
    return "does-not-fit"


def discover_local(
    roots: list[Path], node_id: str, node: dict[str, Any], peer: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*.gguf"):
                resolved = path.resolve()
                if (
                    resolved not in seen
                    and path.is_file()
                    and _has_gguf_magic(path)
                ):
                    seen.add(resolved)
                    files.append(path)
        except OSError:
            continue

    groups: dict[str, list[Path]] = {}
    projectors = {
        path for path in files if path.name.lower().startswith("mmproj")
    }
    for path in files:
        if path in projectors:
            continue
        match = SHARD.match(path.name)
        key = str(path.parent / ((match.group(1) if match else path.stem).lower()))
        groups.setdefault(key, []).append(path)

    records: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    for shards in groups.values():
        shards.sort()
        if not _complete_shard_set(shards):
            continue
        try:
            total_size = sum(path.stat().st_size for path in shards)
        except OSError:
            continue
        first = shards[0]
        match = SHARD.match(first.name)
        display = match.group(1) if match else first.stem
        metadata = read_metadata(first)
        model_id = _model_id(shards, total_size)
        lower = display.lower()
        projector = next((path for path in projectors if path.parent == first.parent), None)
        record: dict[str, Any] = {
            "id": model_id,
            "name": display,
            "architecture": metadata.get("architecture", "unknown"),
            "quantization": _quantization(display),
            "sizeBytes": total_size,
            "contextLength": max(512, int(metadata.get("contextLength") or 4096)),
            "capability": "vision" if projector or any(x in lower for x in ("vision", "-vl", "_vl")) else "text",
            "shards": len(shards),
            "locations": [{"nodeId": node_id, "path": str(first), "source": _source(first)}],
            "fit": _fit(total_size, node, peer, metadata),
            "remoteOnly": False,
            "mtp": any(has_nextn_tensors(shard) for shard in shards),
        }
        for key in ("layerCount", "embeddingLength", "attentionHeadCount", "attentionHeadCountKv"):
            if key in metadata:
                record[key] = metadata[key]
        if projector:
            record["projector"] = str(projector)
        records.append(record)
        paths[model_id] = str(first)
    records.sort(key=lambda value: value["name"].lower())
    return records, paths


def _source(path: Path) -> str:
    return "lm-studio" if ".lmstudio" in {part.lower() for part in path.parts} else "custom"


def refresh_fits(
    models: list[dict[str, Any]], node: dict[str, Any], peer: dict[str, Any] | None
) -> None:
    for model in models:
        model["fit"] = _fit(int(model["sizeBytes"]), node, peer, model)


def _has_gguf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _complete_shard_set(shards: list[Path]) -> bool:
    matches = [SHARD.match(path.name) for path in shards]
    if not any(matches):
        return len(shards) == 1
    if not all(matches):
        return False
    totals = {int(match.group(3)) for match in matches if match}
    indices = sorted(int(match.group(2)) for match in matches if match)
    return len(totals) == 1 and indices == list(range(1, next(iter(totals)) + 1))


def merge_remote(local: list[dict[str, Any]], remote: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {model["id"]: dict(model) for model in local}
    for incoming in remote:
        model = dict(incoming)
        existing = merged.get(model["id"])
        if existing:
            locations = list(existing.get("locations", []))
            for location in model.get("locations", []):
                if location not in locations:
                    locations.append(location)
            existing["locations"] = locations
            continue
        model["remoteOnly"] = True
        merged[model["id"]] = model
    return sorted(merged.values(), key=lambda value: value["name"].lower())
