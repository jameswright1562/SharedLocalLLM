from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sharedlocalllm_backend.autotune import (
    Autotuner,
    _run_bench_process,
    allocations_from_split,
    apply_tune_result,
    build_bench_args,
    candidate_gpu_layer_totals,
    candidate_thread_counts,
    default_tune_state,
    generation_rows,
    locate_bench,
    parse_bench_output,
    plan_stages,
    prompt_rows,
    select_winner,
    split_fractions,
    topology_fingerprint,
)
from sharedlocalllm_backend.errors import BackendError


def row(**overrides):
    values = {
        "n_prompt": 512, "n_gen": 0, "n_batch": 512, "n_ubatch": 256,
        "n_gpu_layers": 32, "n_threads": 8, "tensor_split": "50/50",
        "type_k": "f16", "type_v": "f16", "no_op_offload": 0, "poll": 50,
        "avg_ts": 100.0, "stddev_ts": 1.0,
    }
    values.update(overrides)
    return values


def test_locate_bench_finds_the_pinned_executable(tmp_path: Path) -> None:
    (tmp_path / "llama-bench.exe").touch()
    assert locate_bench([tmp_path, Path("elsewhere")]) == tmp_path / "llama-bench.exe"
    assert locate_bench([Path("missing")]) is None


def test_build_bench_args_targets_model_rpc_and_json_output() -> None:
    state = default_tune_state(total_layers=32, cores=8)
    args = build_bench_args(
        "model.gguf", rpc_endpoint="127.0.0.1:50052",
        prompt_tokens=20000, state=state, repetitions=2,
    )

    assert args[args.index("-m") + 1] == "model.gguf"
    assert args[args.index("-rpc") + 1] == "127.0.0.1:50052"
    assert args[args.index("-p") + 1] == "20000"
    assert args[args.index("-r") + 1] == "2"
    assert args[args.index("-o") + 1] == "json"
    assert "-fa" not in args or args[args.index("-fa") + 1] == "on"


def test_build_bench_args_omit_rpc_without_a_worker() -> None:
    args = build_bench_args(
        "model.gguf", prompt_tokens=512,
        state=default_tune_state(32, 8),
    )
    assert "-rpc" not in args


def test_stage_overrides_win_over_current_state() -> None:
    state = default_tune_state(32, 8)
    args = build_bench_args(
        "model.gguf", prompt_tokens=512, state=state,
        overrides={"batch": "1024,2048", "ubatch": "512"},
    )
    assert args[args.index("-b") + 1] == "1024,2048"
    assert args[args.index("-ub") + 1] == "512"


def test_tensor_split_kv_poll_and_op_offload_flags() -> None:
    state = default_tune_state(32, 8)
    args = build_bench_args(
        "model.gguf", prompt_tokens=512, state=state,
        overrides={
            "tensor_split": "40/60,45/55", "k_types": "q4_0,q8_0",
            "v_types": "q4_0,q8_0", "op_offload": "0,1", "poll": "25,50",
        },
    )
    assert args[args.index("-ts") + 1] == "40/60,45/55"
    assert args[args.index("-ctk") + 1] == "q4_0,q8_0"
    assert args[args.index("-ctv") + 1] == "q4_0,q8_0"
    assert args[args.index("-nopo") + 1] == "0,1"
    assert args[args.index("--poll") + 1] == "25,50"


def test_unset_optionals_are_left_out() -> None:
    state = default_tune_state(32, 8)
    state["op_offload"] = None
    state["poll"] = None
    state["tensor_split"] = None
    args = build_bench_args("model.gguf", prompt_tokens=512, state=state)
    for flag in ("-ts", "--poll"):
        assert flag not in args
    # Default KV cache type still needs an explicit value.
    assert args[args.index("-ctk") + 1] == "f16"


def test_batch_groups_never_pair_a_ubatch_above_its_batch() -> None:
    stages = [stage for stage in plan_stages("full", 32, 8) if stage["id"] == "batch"]
    assert stages
    for stage in stages:
        for batch in str(stage["overrides"]["batch"]).split(","):
            assert int(batch) >= int(stage["overrides"]["ubatch"])


def test_plan_stages_quick_is_a_subset_of_full_and_ends_with_verification() -> None:
    quick = plan_stages("quick", 32, 8)
    full = plan_stages("full", 32, 8)

    assert quick[-1]["id"] == "final-verification"
    assert full[-1]["id"] == "final-verification"
    quick_ids = {stage["id"] for stage in quick}
    full_ids = {stage["id"] for stage in full}
    assert {"tensor-split", "kv-cache", "op-offload", "poll"} <= full_ids
    assert not {"tensor-split", "kv-cache", "op-offload", "poll"} & quick_ids
    assert quick_ids <= full_ids


def test_plan_stages_skips_gpu_layers_without_layer_metadata() -> None:
    stages = plan_stages("full", 0, 8)
    assert all(stage["id"] != "gpu-layers" for stage in stages)


def test_candidates_are_derived_from_detected_hardware() -> None:
    assert candidate_thread_counts(8) == [2, 4, 6, 8]
    assert candidate_thread_counts(2) == [1, 2]
    assert candidate_thread_counts(1) == [1]
    assert candidate_gpu_layer_totals(32) == [16, 24, 32]
    assert candidate_gpu_layer_totals(3) == [1, 2, 3]
    assert candidate_gpu_layer_totals(0) == []


def test_parse_bench_output_accepts_array_or_single_object() -> None:
    assert parse_bench_output('[{"avg_ts": 1.0}]') == [{"avg_ts": 1.0}]
    assert parse_bench_output('{"avg_ts": 2.0}') == [{"avg_ts": 2.0}]
    with pytest.raises(BackendError):
        parse_bench_output("not json")


def test_winner_selection_uses_prompt_rows_only() -> None:
    rows = [
        row(avg_ts=10.0),
        row(n_prompt=0, n_gen=128, avg_ts=9999.0),
        row(avg_ts=30.0),
    ]
    assert len(prompt_rows(rows)) == 2
    assert len(generation_rows(rows)) == 1
    winner = select_winner(prompt_rows(rows))
    assert winner and winner["avg_ts"] == 30.0
    assert select_winner([]) is None


def test_split_fractions_parses_slash_lists() -> None:
    assert split_fractions("55/45") == [55.0, 45.0]
    assert split_fractions("") == []


def test_allocations_follow_the_tuned_split_with_remainder_on_the_last_node() -> None:
    nodes = [
        {"id": "local", "gpu": {"vramAvailableGb": 8}},
        {"id": "peer", "gpu": {"vramAvailableGb": 8}},
    ]
    allocations = allocations_from_split([60.0, 40.0], nodes, 10)

    assert allocations == [
        {"nodeId": "local", "layers": 6, "kind": "gpu"},
        {"nodeId": "peer", "layers": 4, "kind": "gpu"},
    ]


def test_topology_fingerprint_tracks_vram_peer_and_device_changes() -> None:
    base = [{"id": "a", "name": "PC1", "online": True,
             "ramAvailableGb": 16.0, "gpu": {"name": "RTX", "vramAvailableGb": 8.0}}]
    peer = base + [{"id": "b", "name": "PC2", "online": True,
                    "ramAvailableGb": 8.0, "gpu": {"name": "Laptop GPU", "vramAvailableGb": 6.0}}]

    assert topology_fingerprint(base) == topology_fingerprint(list(reversed(base)))
    assert topology_fingerprint(base) != topology_fingerprint(peer)
    changed = [{**base[0], "gpu": {"name": "RTX", "vramAvailableGb": 12.0}}]
    assert topology_fingerprint(base) != topology_fingerprint(changed)


def test_apply_tune_result_merges_winners_and_keeps_context() -> None:
    tune = {
        "winners": {
            "batchSize": 2048, "uBatch": 512, "cpuThreads": 8,
            "kvCacheK": "q4_0", "kvCacheV": "q8_0",
            "gpuLayersAllocations": [{"nodeId": "local", "layers": 20, "kind": "gpu"}],
        }
    }
    config = apply_tune_result({"contextSize": 16384, "flashAttention": True}, tune)

    assert config["contextSize"] == 16384
    assert config["flashAttention"] is True
    assert config["batchSize"] == 2048
    assert config["uBatch"] == 512
    assert config["cpuThreads"] == 8
    assert config["kvCacheK"] == "q4_0"
    assert config["kvCacheV"] == "q8_0"
    assert config["gpuLayers"] == [{"nodeId": "local", "layers": 20, "kind": "gpu"}]


def test_apply_tune_result_without_winners_returns_an_equivalent_config() -> None:
    config = {"contextSize": 4096}
    assert apply_tune_result(config, {"winners": {}}) == config


def _make_runner(delays: float = 0.0):
    calls: list[list[str]] = []

    def flag_value(args: list[str], flag: str) -> str | None:
        return args[args.index(flag) + 1] if flag in args else None

    async def runner(exe: Path, args: list[str]) -> list[dict]:
        calls.append(args)
        await asyncio.sleep(delays)

        generated = int(flag_value(args, "-n") or 0)
        if generated:
            return [
                row(avg_ts=95.0),
                row(n_prompt=0, n_gen=generated, avg_ts=12.5),
            ]
        threads = flag_value(args, "-t")
        if threads and "," in threads:
            return [row(n_threads=int(v), avg_ts=float(v) * 10.0 + 10.0)
                    for v in threads.split(",")]
        layers = flag_value(args, "-ngl")
        if layers and "," in layers:
            return [row(n_gpu_layers=int(v), avg_ts=float(v) + 40.0)
                    for v in layers.split(",")]
        batch = flag_value(args, "-b")
        if batch and "," in batch:
            rates = {"512": 108.0, "1024": 112.0, "2048": 116.0}
            ubatch_rates = {"256": 5.0, "512": 7.0, "1024": 2.0, "2048": 0.0}
            ubatch = flag_value(args, "-ub") or "256"
            return [row(n_batch=int(v), n_ubatch=int(ubatch),
                        avg_ts=rates.get(v, 100.0) + ubatch_rates.get(ubatch, 0.0))
                    for v in batch.split(",")]
        return [row(avg_ts=95.0)]

    return runner, calls


def test_autotuner_runs_quick_sweep_and_reports_winners() -> None:
    runner, calls = _make_runner()
    tuner = Autotuner(runner=runner)
    completed: list[dict] = []

    async def scenario() -> dict:
        started = await tuner.start(
            exe=Path("llama-bench.exe"), model_path="m.gguf",
            model_id="model-1", model_name="Model One",
            depth="quick", nodes=[], total_layers=32, cores=8,
            on_complete=completed.append,
        )
        while started["status"] == "running":
            await asyncio.sleep(0)
            started = tuner.status()
        return started

    status = asyncio.run(scenario())

    assert status["status"] == "complete"
    assert completed and completed[0]["modelId"] == "model-1"
    result = status["result"]
    winners = result["winners"]
    assert winners["batchSize"] == 2048
    assert winners["uBatch"] == 512
    assert winners["gpuLayers"] == 32
    assert winners["cpuThreads"] == 8
    assert result["promptTokensPerSecond"] == 95.0
    assert calls and calls[0][:2] == ["-m", "m.gguf"]


def test_autotuner_status_is_idle_before_start_and_running_during() -> None:
    runner, _calls = _make_runner()
    gate = asyncio.Event()

    async def slow_runner(exe: Path, args: list[str]) -> list[dict]:
        await gate.wait()
        return []

    tuner = Autotuner(runner=slow_runner)

    async def scenario() -> tuple[str, str]:
        initial = tuner.status()["status"]
        await tuner.start(
            exe=Path("llama-bench.exe"), model_path="m.gguf",
            model_id="m", model_name="M", depth="quick",
            nodes=[], total_layers=32, cores=8,
        )
        running = tuner.status()
        gate.set()
        while tuner.status()["status"] == "running":
            await asyncio.sleep(0)
        return initial, running["status"]

    initial, running = asyncio.run(scenario())
    assert initial == "idle"
    assert running == "running"


def test_autotuner_cancel_stops_the_sweep() -> None:
    runner, _calls = _make_runner()
    release: list[asyncio.Event] = []

    async def blocking_runner(exe: Path, args: list[str]) -> list[dict]:
        event = asyncio.Event()
        release.append(event)
        await event.wait()
        return []

    tuner = Autotuner(runner=blocking_runner)

    async def scenario() -> str:
        await tuner.start(
            exe=Path("llama-bench.exe"), model_path="m.gguf",
            model_id="m", model_name="M", depth="quick",
            nodes=[], total_layers=32, cores=8,
        )
        await asyncio.sleep(0)
        await tuner.cancel()
        while tuner.status()["status"] == "running":
            await asyncio.sleep(0)
        return tuner.status()["status"]

    assert asyncio.run(scenario()) == "cancelled"


def test_autotuner_resets_stage_winners_between_runs() -> None:
    tuner = Autotuner()

    async def run_with_rate(rate: float, batch: int) -> dict:
        async def runner(_exe: Path, args: list[str]) -> list[dict]:
            generated = int(args[args.index("-n") + 1])
            if generated:
                return [
                    row(avg_ts=rate),
                    row(n_prompt=0, n_gen=generated, avg_ts=rate),
                ]
            return [row(n_batch=batch, n_ubatch=min(batch, 256), avg_ts=rate)]

        tuner._runner = runner
        await tuner.start(
            exe=Path("llama-bench.exe"), model_path="m.gguf",
            model_id="m", model_name="M", depth="quick",
            nodes=[], total_layers=32, cores=8,
        )
        while tuner.status()["status"] == "running":
            await asyncio.sleep(0)
        return tuner.status()["result"]

    async def scenario() -> tuple[dict, dict]:
        first = await run_with_rate(100.0, 2048)
        second = await run_with_rate(10.0, 512)
        return first, second

    first, second = asyncio.run(scenario())
    assert first["winners"]["batchSize"] == 2048
    assert second["winners"]["batchSize"] == 512


def test_cancelled_bench_process_is_terminated(monkeypatch) -> None:
    class Process:
        returncode = None

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        async def communicate(self):
            await asyncio.Event().wait()

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.returncode = -1
            return self.returncode

    process = Process()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr("sharedlocalllm_backend.autotune.asyncio.create_subprocess_exec", create_process)

    async def scenario() -> None:
        task = asyncio.create_task(_run_bench_process(Path("llama-bench.exe"), ["-m", "m.gguf"]))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert process.terminated is True
    assert process.killed is False


def test_failed_bench_process_redacts_the_model_path(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "private" / "model.gguf"

    class Process:
        returncode = 1

        async def communicate(self):
            return b"", f"failed to load {model_path}".encode()

    async def create_process(*_args, **_kwargs):
        return Process()

    monkeypatch.setattr("sharedlocalllm_backend.autotune.asyncio.create_subprocess_exec", create_process)

    with pytest.raises(BackendError) as caught:
        asyncio.run(
            _run_bench_process(
                Path("llama-bench.exe"), ["-m", str(model_path)],
            )
        )

    assert str(model_path) not in caught.value.message
    assert "<model-path>" in caught.value.message


def test_autotuner_refuses_to_run_twice() -> None:
    gate = asyncio.Event()

    async def slow_runner(exe: Path, args: list[str]) -> list[dict]:
        await gate.wait()
        return []

    tuner = Autotuner(runner=slow_runner)

    async def scenario() -> BackendError:
        try:
            await tuner.start(
                exe=Path("llama-bench.exe"), model_path="m.gguf",
                model_id="m", model_name="M", depth="quick",
                nodes=[], total_layers=32, cores=8,
            )
            await tuner.start(
                exe=Path("llama-bench.exe"), model_path="m.gguf",
                model_id="m2", model_name="M2", depth="quick",
                nodes=[], total_layers=32, cores=8,
            )
        except BackendError as error:
            gate.set()
            while tuner.status()["status"] == "running":
                await asyncio.sleep(0)
            return error
        raise AssertionError("second start did not raise")

    error = asyncio.run(scenario())
    assert "already" in error.message.lower()


def test_failed_stage_marks_the_session_failed() -> None:
    async def failing_runner(exe: Path, args: list[str]) -> list[dict]:
        raise BackendError("autotune_stage_failed", "boom")

    tuner = Autotuner(runner=failing_runner)

    async def scenario() -> dict:
        await tuner.start(
            exe=Path("llama-bench.exe"), model_path="m.gguf",
            model_id="m", model_name="M", depth="quick",
            nodes=[], total_layers=32, cores=8,
        )
        while tuner.status()["status"] in ("running", "idle"):
            await asyncio.sleep(0)
        return tuner.status()

    status = asyncio.run(scenario())
    assert status["status"] == "failed"
