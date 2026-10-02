"""Load stages and the metadata preserved throughout cluster startup."""

from typing import Any, Literal

LoadStage = Literal["preparing", "tunnel", "staging", "loading_weights"]
EngineName = Literal["builtin", "llama-server"]


def loading_cluster(
    model_id: str, coordinator_id: str, worker_id: str | None,
    engine: EngineName, expected_bytes: int,
) -> dict[str, Any]:
    return {
        "status": "loading", "modelId": model_id,
        "coordinatorNodeId": coordinator_id, "workerNodeId": worker_id,
        "engine": engine, "stage": "preparing", "expectedBytes": expected_bytes,
    }
