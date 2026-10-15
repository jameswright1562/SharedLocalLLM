"""Load stages and the metadata preserved throughout cluster startup."""

from typing import Any, Literal

LoadStage = Literal["preparing", "tunnel", "staging", "loading_weights"]
EngineName = Literal["builtin", "llama-server"]


def loading_cluster(
    model_id: str, coordinator_id: str, worker_id: str | None,
    engine: EngineName, expected_bytes: int,
    worker_ids: list[str] | None = None,
) -> dict[str, Any]:
    workers = list(worker_ids) if worker_ids else ([worker_id] if worker_id else [])
    return {
        "status": "loading", "modelId": model_id,
        "coordinatorNodeId": coordinator_id, "workerNodeId": workers[0] if workers else None,
        "workerNodeIds": workers or None,
        "engine": engine, "stage": "preparing", "expectedBytes": expected_bytes,
    }


def cluster_workers(cluster: dict[str, Any]) -> list[str]:
    """All worker ids, combining the multi-node list with the legacy single field."""
    ids = list(cluster.get("workerNodeIds") or [])
    single = cluster.get("workerNodeId")
    if single and single not in ids:
        ids.insert(0, single)
    return ids
