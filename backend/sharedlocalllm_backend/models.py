from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .gguf import read_metadata

SHARD = re.compile(r"^(.*?)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
QUANT = re.compile(r"(?:^|[-_.])(Q\d(?:_[A-Z0-9]+)+|Q\d_[A-Z0-9]+)(?:[-_.]|$)", re.IGNORECASE)


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


def _model_id(name: str, size: int) -> str:
    return hashlib.sha256(f"{name.lower()}|{size}".encode()).hexdigest()[:24]


def _quantization(name: str) -> str:
    match = QUANT.search(name.upper())
    return match.group(1).upper() if match else "unknown"


def _fit(size: int, node: dict[str, Any], peer: dict[str, Any] | None) -> str:
    local_vram = float(node.get("gpu", {}).get("vramAvailableGb", 0)) * 1024**3
    local_ram = float(node.get("ramAvailableGb", 0)) * 1024**3
    if size <= local_vram:
        return "single-node"
    peer_vram = 0.0
    if peer:
        peer_vram = float(peer.get("gpu", {}).get("vramAvailableGb", 0)) * 1024**3
    if peer_vram and size <= local_vram + peer_vram:
        return "combined-gpu"
    if size <= local_vram + local_ram:
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
                if resolved not in seen and path.is_file():
                    seen.add(resolved)
                    files.append(path)
        except OSError:
            continue

    groups: dict[str, list[Path]] = {}
    for path in files:
        match = SHARD.match(path.name)
        key = str(path.parent / (match.group(1) if match else path.stem))
        groups.setdefault(key, []).append(path)

    records: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    for shards in groups.values():
        shards.sort()
        total_size = sum(path.stat().st_size for path in shards)
        first = shards[0]
        match = SHARD.match(first.name)
        display = match.group(1) if match else first.stem
        metadata = read_metadata(first)
        model_id = _model_id(display, total_size)
        lower = display.lower()
        record: dict[str, Any] = {
            "id": model_id,
            "name": display,
            "architecture": metadata.get("architecture", "unknown"),
            "quantization": _quantization(display),
            "sizeBytes": total_size,
            "contextLength": int(metadata.get("contextLength", 4096)),
            "capability": "vision" if any(x in lower for x in ("vision", "-vl", "_vl")) else "text",
            "shards": len(shards),
            "locations": [{"nodeId": node_id, "path": str(first), "source": _source(first)}],
            "fit": _fit(total_size, node, peer),
            "remoteOnly": False,
        }
        for key in ("layerCount", "embeddingLength", "attentionHeadCount", "attentionHeadCountKv"):
            if key in metadata:
                record[key] = metadata[key]
        records.append(record)
        paths[model_id] = str(first)
    records.sort(key=lambda value: value["name"].lower())
    return records, paths


def _source(path: Path) -> str:
    return "lm-studio" if ".lmstudio" in {part.lower() for part in path.parts} else "custom"


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
