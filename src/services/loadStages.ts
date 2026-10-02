import type { ClusterSession } from "../types";

const LOAD_STAGES = [
  { id: "preparing", label: "Preparing allocation" },
  { id: "tunnel", label: "Opening worker tunnel" },
  { id: "staging", label: "Preparing compute devices" },
  { id: "loading_weights", label: "Loading weights" },
] as const;

export type LoadStage = (typeof LOAD_STAGES)[number]["id"];

export function usesWorker(cluster: ClusterSession): boolean {
  return Boolean(
    cluster.workerNodeId || cluster.bytesToWorker !== undefined || (cluster.expectedBytes ?? 0) > 0,
  );
}

export function loadStagesFor(cluster: ClusterSession): Array<{ id: LoadStage; label: string }> {
  const remote = usesWorker(cluster);
  return LOAD_STAGES.filter((stage) =>
    stage.id === "tunnel" ? remote : stage.id !== "staging" || cluster.engine !== "llama-server",
  ).map((stage) => ({
    ...stage,
    label:
      stage.id === "staging" && remote
        ? "Staging worker devices"
        : stage.id === "loading_weights" && cluster.engine === "llama-server"
          ? "Loading weights and checking readiness"
          : stage.label,
  }));
}
