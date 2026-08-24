"""Staged llama-bench autotune bound to per-model load configurations.

Drives the pinned ``llama-bench.exe`` (verified installer output, never a
downloaded ad-hoc binary) through a staged sweep — batch/ubatch groups, GPU
layer totals, CPU threads and, in the full profile, tensor split, KV cache
types, operation offload, and RPC polling. Each stage keeps its winner as the
baseline for the next stage, mirroring a coordinate-descent search, and a
final prompt + generation verification measures the combined configuration.

Tuning is experimental: results are measured for one hardware/link topology
and are only re-applied while that topology fingerprint still matches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from .errors import BackendError
from .server_engine import _redact_server_output

BENCH_EXECUTABLE = "llama-bench.exe"

BATCH_GROUPS: tuple[tuple[str, str], ...] = (
    ("512,1024,2048", "256"),
    ("512,1024,2048", "512"),
    ("1024,2048", "1024"),
    ("2048", "2048"),
)
TENSOR_SPLITS = "40/60,45/55,50/50,55/45,60/40"
KV_TYPES = "q4_0,q8_0"
OP_OFFLOAD_VALUES = "0,1"
POLL_VALUES = "0,25,50,75,100"
QUICK_ONLY_STAGE_IDS = {"batch", "gpu-layers", "threads"}
FULL_EXTRA_STAGE_IDS = {"tensor-split", "kv-cache", "op-offload", "poll"}

BenchRunner = Callable[[Path, list[str]], Awaitable[list[dict[str, Any]]]]
FinishCallback = Callable[[dict[str, Any]], Awaitable[None]]


def locate_bench(search_dirs: list[Path] | tuple[Path, ...]) -> Path | None:
    """First directory that contains the pinned llama-bench executable."""
    for directory in search_dirs:
        candidate = Path(directory) / BENCH_EXECUTABLE
        if candidate.is_file():
            return candidate
    return None


def candidate_thread_counts(cores: int) -> list[int]:
    """Quarter/half/three-quarter/full thread counts from detected cores."""
    safe = max(1, int(cores))
    values = sorted({max(1, safe // 4), max(1, safe // 2), max(1, 3 * safe // 4), safe})
    return values


def candidate_gpu_layer_totals(total_layers: int) -> list[int]:
    """Half/three-quarter/full offload totals bounded by the model's layers."""
    if total_layers <= 0:
        return []
    safe = int(total_layers)
    return sorted({max(1, safe // 2), max(1, 3 * safe // 4), safe})


def default_tune_state(total_layers: int, cores: int) -> dict[str, Any]:
    """Baseline configuration every stage starts from."""
    return {
        "batch": None,
        "ubatch": None,
        "gpu_layers": str(total_layers) if total_layers > 0 else "99",
        "threads": str(max(1, cores // 2) if cores else 4),
        "k_type": "f16",
        "v_type": "f16",
        "tensor_split": None,
        "op_offload": None,
        "poll": None,
    }


def plan_stages(depth: str, total_layers: int, cores: int) -> list[dict[str, Any]]:
    """Ordered stage plan; each entry holds llama-bench value lists."""
    stages: list[dict[str, Any]] = []
    for batch, ubatch in BATCH_GROUPS:
        stages.append({
            "id": "batch", "label": f"batch {batch} at ubatch {ubatch}",
            "overrides": {"batch": batch, "ubatch": ubatch},
        })
    layer_totals = candidate_gpu_layer_totals(total_layers)
    if layer_totals:
        stages.append({
            "id": "gpu-layers", "label": "GPU layer total",
            "overrides": {"gpu_layers": ",".join(str(v) for v in layer_totals)},
        })
    threads = candidate_thread_counts(cores)
    stages.append({
        "id": "threads", "label": "CPU threads",
        "overrides": {"threads": ",".join(str(v) for v in threads)},
    })
    if depth == "full":
        stages.extend([
            {"id": "tensor-split", "label": "RPC tensor split",
             "overrides": {"tensor_split": TENSOR_SPLITS}},
            {"id": "kv-cache", "label": "KV cache types",
             "overrides": {"k_types": KV_TYPES, "v_types": KV_TYPES}},
            {"id": "op-offload", "label": "Operation offload",
             "overrides": {"op_offload": OP_OFFLOAD_VALUES}},
            {"id": "poll", "label": "RPC polling", "overrides": {"poll": POLL_VALUES}},
        ])
    stages.append({
        "id": "final-verification", "label": "Final verification",
        "overrides": {}, "gen_tokens": 256,
    })
    return stages


def build_bench_args(
    model_path: str,
    *,
    rpc_endpoint: str | None = None,
    prompt_tokens: int,
    state: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    gen_tokens: int = 0,
    depth: int = 0,
    repetitions: int = 2,
) -> list[str]:
    """One llama-bench invocation; comma lists make llama-bench cross-product."""
    merged = {key: state.get(key) for key in state}
    for key, value in (overrides or {}).items():
        merged[key] = value

    args = [
        "-m", model_path,
        "-p", str(int(prompt_tokens)),
        "-n", str(int(gen_tokens)),
        "-d", str(int(depth)),
        "-ngl", str(merged["gpu_layers"]),
        "-t", str(merged["threads"]),
        "-ctk", str(merged["k_type"] if merged.get("k_types") is None else merged["k_types"]),
        "-ctv", str(merged["v_type"] if merged.get("v_types") is None else merged["v_types"]),
        "-sm", "layer",
        "-fa", "on",
        "-r", str(int(repetitions)),
        "-o", "json",
    ]
    if merged.get("batch") is not None and merged.get("ubatch") is not None:
        args += ["-b", str(merged["batch"]), "-ub", str(merged["ubatch"])]
    if merged.get("tensor_split") is not None:
        args += ["-ts", str(merged["tensor_split"])]
    if merged.get("op_offload") is not None:
        args += ["-nopo", str(merged["op_offload"])]
    if merged.get("poll") is not None:
        args += ["--poll", str(merged["poll"])]
    if rpc_endpoint:
        args += ["-rpc", rpc_endpoint]
    return args


def parse_bench_output(text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise BackendError(
            "autotune_output_invalid",
            "llama-bench did not produce readable JSON output.",
            "Retry the tuning run; if it repeats, report it with the backend log.",
        ) from error
    rows = parsed if isinstance(parsed, list) else [parsed]
    return [row for row in rows if isinstance(row, dict)]


def prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _count(row, "n_prompt") > 0 and _count(row, "n_gen") == 0]


def generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _count(row, "n_prompt") == 0 and _count(row, "n_gen") > 0]


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fastest row by average tokens per second, or None without results."""
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("avg_ts") or 0.0))


def split_fractions(value: str | None) -> list[float]:
    """"55/45" style llama.cpp splits as per-device percentage numbers."""
    if not value or not str(value).strip():
        return []
    return [float(part) for part in str(value).split("/")]


def allocations_from_split(
    fractions: list[float], nodes: list[dict[str, Any]], gpu_layers: int,
) -> list[dict[str, Any]]:
    """Per-node GPU layer allocations proportional to a tuned split.

    Remainder layers land on the last node so the total always matches.
    """
    capable = [
        node for node in nodes
        if float((node.get("gpu") or {}).get("vramAvailableGb", 0)) > 0
    ]
    usable = fractions[:len(capable)] if capable else []
    if not usable or sum(usable) <= 0:
        return []
    total = float(sum(usable))
    allocations: list[dict[str, Any]] = []
    assigned = 0
    for index, fraction in enumerate(usable):
        if index == len(usable) - 1:
            layers = int(gpu_layers) - assigned
        else:
            layers = int(int(gpu_layers) * fraction / total)
        assigned += layers
        allocations.append({"nodeId": capable[index]["id"], "layers": layers, "kind": "gpu"})
    return [allocation for allocation in allocations if allocation["layers"] > 0]


def apply_tune_result(load_config: dict[str, Any], tune: dict[str, Any]) -> dict[str, Any]:
    """Merge tuned winners into a load configuration without dropping fields.

    Context size stays user-owned: a winner measured at the tuning prompt depth
    must never silently shrink or grow the real server context.
    """
    winners = tune.get("winners") or {}
    config = dict(load_config)
    mapping = {
        "batchSize": "batchSize", "uBatch": "uBatch", "cpuThreads": "cpuThreads",
        "kvCacheK": "kvCacheK", "kvCacheV": "kvCacheV",
        "noOpOffload": "noOpOffload", "poll": "rpcPoll",
    }
    for winner_key, config_key in mapping.items():
        value = winners.get(winner_key)
        if value is not None:
            config[config_key] = value
    allocations = winners.get("gpuLayersAllocations")
    if allocations:
        config["gpuLayers"] = allocations
    return config


def topology_fingerprint(nodes: list[dict[str, Any]]) -> str:
    """Stable hash of the compute topology a tuning result was measured on."""
    relevant = []
    for node in nodes:
        gpu = node.get("gpu") or {}
        relevant.append({
            "id": node.get("id"),
            "name": node.get("name"),
            "online": bool(node.get("online", True)),
            "ramAvailableGb": round(float(node.get("ramAvailableGb", 0)), 1),
            "gpuName": gpu.get("name"),
            "vramAvailableGb": round(float(gpu.get("vramAvailableGb", 0)), 1),
        })
    payload = json.dumps(sorted(relevant, key=lambda item: str(item["id"])), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Autotuner:
    """Single-flight orchestrator around llama-bench with status polling."""

    def __init__(self, store: Any = None, runner: BenchRunner | None = None) -> None:
        self.store = store
        self._runner = runner or _run_bench_process
        self._task: asyncio.Task[None] | None = None
        self._status: dict[str, Any] = {"status": "idle"}
        self._events: list[dict[str, Any]] = []
        self._stage_best: dict[str, float] = {}

    def status(self) -> dict[str, Any]:
        snapshot = dict(self._status)
        snapshot["events"] = list(self._events[-20:])
        return snapshot

    async def start(
        self,
        *,
        exe: Path,
        model_path: str,
        model_id: str,
        model_name: str,
        depth: str,
        nodes: list[dict[str, Any]],
        total_layers: int,
        cores: int | None = None,
        rpc_endpoint: str | None = None,
        prompt_tokens: int = 4096,
        fingerprint: str | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
        on_finish: FinishCallback | None = None,
    ) -> dict[str, Any]:
        if self._task and not self._task.done():
            raise BackendError(
                "autotune_already_running",
                "A tuning run is already in progress.",
                "Wait for it to finish or cancel it first.",
            )
        resolved_depth = "full" if depth == "full" else "quick"
        self._events = []
        self._stage_best = {}
        self._status = {
            "status": "running", "modelId": model_id, "modelName": model_name,
            "depth": resolved_depth, "stageIndex": 0,
            "stageCount": len(plan_stages(resolved_depth, total_layers, cores if cores else os.cpu_count() or 8)),
            "currentStage": None, "result": None,
        }
        self._task = asyncio.create_task(self._run(
            exe=exe, model_path=model_path, model_id=model_id, model_name=model_name,
            depth=resolved_depth, nodes=nodes, total_layers=total_layers,
            cores=cores if cores else os.cpu_count() or 8,
            rpc_endpoint=rpc_endpoint, prompt_tokens=prompt_tokens,
            fingerprint=fingerprint or topology_fingerprint(nodes),
            on_complete=on_complete, on_finish=on_finish,
        ))
        return self.status()

    async def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await self._task

    async def _run(
        self,
        *,
        exe: Path,
        model_path: str,
        model_id: str,
        model_name: str,
        depth: str,
        nodes: list[dict[str, Any]],
        total_layers: int,
        cores: int,
        rpc_endpoint: str | None,
        prompt_tokens: int,
        fingerprint: str,
        on_complete: Callable[[dict[str, Any]], None] | None,
        on_finish: FinishCallback | None,
    ) -> None:
        state = default_tune_state(total_layers, cores)
        stages = plan_stages(depth, total_layers, cores)
        rows: list[dict[str, Any]] = []
        try:
            for index, stage in enumerate(stages):
                self._status.update(stageIndex=index + 1, currentStage=stage["label"])
                self._emit({"type": "stage-start", "stage": stage["id"], "label": stage["label"]})
                gen_tokens = int(stage.get("gen_tokens") or 0)
                repetitions = 5 if gen_tokens else 2
                args = build_bench_args(
                    model_path,
                    rpc_endpoint=rpc_endpoint,
                    prompt_tokens=prompt_tokens,
                    state=state,
                    overrides=stage["overrides"],
                    gen_tokens=gen_tokens,
                    repetitions=repetitions,
                )
                rows = await self._runner(exe, args)
                wanted = generation_rows if gen_tokens else prompt_rows
                winner = select_winner(wanted(rows))
                rate = round(float(winner["avg_ts"]), 2) if winner else 0.0
                if stage["id"] != "final-verification" and winner:
                    previous = self._stage_best.get(stage["id"])
                    if previous is None or rate > previous:
                        self._stage_best[stage["id"]] = rate
                        self._absorb_winner(state, stage["id"], winner)
                self._emit({
                    "type": "stage-result", "stage": stage["id"],
                    "bestTokensPerSecond": rate,
                })
                if self.store:
                    self.store.log(
                        "INFO", "autotune_stage",
                        f"{stage['id']} winner {rate} tok/s",
                    )
            final_prompt = select_winner(prompt_rows(rows))
            final_generation = select_winner(generation_rows(rows))
            result = {
                "modelId": model_id,
                "modelName": model_name,
                "depth": depth,
                "ranAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fingerprint": fingerprint,
                "winners": self._winners(state, nodes),
                "promptTokensPerSecond": round(float(final_prompt["avg_ts"]), 2) if final_prompt else 0.0,
                "generationTokensPerSecond": round(float(final_generation["avg_ts"]), 2) if final_generation else 0.0,
            }
            self._status.update(status="complete", result=result, currentStage=None)
            if on_complete:
                on_complete(result)
        except asyncio.CancelledError:
            self._status.update(status="cancelled", currentStage=None)
        except BackendError as error:
            self._status.update(status="failed", error=error.message, currentStage=None)
        except Exception as error:  # noqa: BLE001 - surfaced through status polling
            self._status.update(status="failed", error=str(error), currentStage=None)
        finally:
            if on_finish:
                await on_finish(self.status())

    @staticmethod
    def _absorb_winner(state: dict[str, Any], stage_id: str, winner: dict[str, Any] | None) -> None:
        if not winner:
            return
        if stage_id == "batch":
            state["batch"] = int(_count(winner, "n_batch"))
            state["ubatch"] = int(_count(winner, "n_ubatch"))
        elif stage_id == "gpu-layers":
            state["gpu_layers"] = str(int(_count(winner, "n_gpu_layers")))
        elif stage_id == "threads":
            state["threads"] = str(int(_count(winner, "n_threads")))
        elif stage_id == "tensor-split":
            state["tensor_split"] = str(winner.get("tensor_split") or "")
        elif stage_id == "kv-cache":
            state["k_type"] = str(winner.get("type_k") or state["k_type"])
            state["v_type"] = str(winner.get("type_v") or state["v_type"])
        elif stage_id == "op-offload":
            state["op_offload"] = int(_count(winner, "no_op_offload"))
        elif stage_id == "poll":
            state["poll"] = int(_count(winner, "poll"))

    @staticmethod
    def _winners(state: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, Any]:
        fractions = split_fractions(state["tensor_split"])
        gpu_total = int(state["gpu_layers"]) if str(state["gpu_layers"]).isdigit() else None
        return {
            "batchSize": state["batch"],
            "uBatch": state["ubatch"],
            "cpuThreads": int(state["threads"]) if str(state["threads"]).isdigit() else None,
            "kvCacheK": state["k_type"],
            "kvCacheV": state["v_type"],
            "gpuLayers": gpu_total,
            "tensorSplit": fractions or None,
            "noOpOffload": state["op_offload"],
            "poll": state["poll"],
            "gpuLayersAllocations": (
                allocations_from_split(fractions, nodes, gpu_total)
                if fractions and gpu_total else None
            ),
        }

    def _emit(self, event: dict[str, Any]) -> None:
        self._events.append(event)


def _count(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


async def _run_bench_process(exe: Path, args: list[str]) -> list[dict[str, Any]]:
    """Run the pinned llama-bench once and parse its JSON report."""
    creation_flags = getattr(asyncio.subprocess, "CREATE_NO_WINDOW", 0)
    process = await asyncio.create_subprocess_exec(
        str(exe), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=creation_flags,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        await asyncio.shield(_terminate_bench_process(process))
        raise
    if process.returncode != 0:
        model_path = args[args.index("-m") + 1] if "-m" in args else None
        detail = _redact_server_output(
            stderr.decode("utf-8", errors="replace").strip(),
            model_path,
            None,
        )[-400:]
        raise BackendError(
            "autotune_stage_failed",
            f"llama-bench exited with code {process.returncode}. {detail}".strip(),
            "Check the GPU/RPC setup on both computers, then retry tuning.",
        )
    return parse_bench_output(stdout.decode("utf-8", errors="replace"))


async def _terminate_bench_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 10)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await asyncio.wait_for(process.wait(), 5)
