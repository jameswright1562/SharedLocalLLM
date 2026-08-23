import { useRef, useState } from "react";
import { describeAppError } from "../services/errors";
import { fitLayersByVram } from "../services/splitEstimate";
import type { InferenceBenchmark, PageProps } from "../types";
import { formatRunTime } from "./pageFormat";

export function BenchmarksPage({ snapshot, service, refreshSnapshot, navigate }: PageProps) {
  const [modelId, setModelId] = useState(snapshot.models[0]?.id ?? "");
  const [runs, setRuns] = useState<InferenceBenchmark[]>(snapshot.benchmarks);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const runRef = useRef(0);
  const selectedModel = snapshot.models.find((model) => model.id === modelId);
  const gpuNodes = snapshot.nodes.filter((node) => node.online && node.gpu.vramAvailableGb > 0);
  const plannedSplit = selectedModel ? fitLayersByVram(selectedModel, gpuNodes) : [];
  const activeModelId =
    snapshot.cluster.status === "running"
      ? snapshot.cluster.modelId
      : snapshot.nodes.find((node) => node.clusterStatus === "running")?.clusterModelId;
  const usesRunningInstance = Boolean(activeModelId && modelId === activeModelId);

  async function runBenchmark() {
    if (!modelId || running) return;
    const runId = ++runRef.current;
    setRunning(true);
    setError("");
    try {
      const results = await service.runInferenceBenchmark(modelId);
      if (runRef.current !== runId) return;
      setRuns((current) => [...results, ...current]);
      await refreshSnapshot();
    } catch (reason) {
      if (runRef.current === runId) {
        const detail = describeAppError(reason, "The benchmark could not finish.");
        setError(detail);
        setRuns((current) => [
          {
            id: `failed-${Date.now()}`,
            modelName: selectedModel?.name ?? modelId,
            topology: "local",
            promptTokensPerSecond: 0,
            generationTokensPerSecond: 0,
            loadTimeSeconds: 0,
            memoryPeakGb: 0,
            recommended: false,
            ranAt: new Date().toISOString(),
            error: detail,
          },
          ...current,
        ]);
      }
    } finally {
      if (runRef.current === runId) setRunning(false);
    }
  }

  async function cancelBenchmark() {
    runRef.current += 1;
    setRunning(false);
    setError("");
    try {
      await service.cancelInferenceBenchmark();
    } catch (reason) {
      setError(describeAppError(reason, "The benchmark could not be cancelled."));
    }
  }

  return (
    <div className="page">
      <header className="page-header split-header">
        <div>
          <p className="section-kicker">Placement evidence</p>
          <h1>Performance benchmarks</h1>
          <p>Measured results for this exact model, hardware pair, context, and network route.</p>
        </div>
        <div className="benchmark-actions">
          <label htmlFor="benchmark-model">Benchmark model</label>
          <select
            id="benchmark-model"
            value={modelId}
            disabled={running || snapshot.models.length === 0}
            onChange={(event) => setModelId(event.target.value)}
          >
            {snapshot.models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
          {usesRunningInstance ? (
            <p className="benchmark-split" role="status">
              {selectedModel?.name} is running — benchmarks the loaded instance without reloading.
            </p>
          ) : (
            plannedSplit.length > 0 && (
              <p className="benchmark-split" role="status">
                Automatic GPU split:{" "}
                {plannedSplit
                  .map((allocation) => {
                    const name = gpuNodes.find((node) => node.id === allocation.nodeId)?.name;
                    return `${name ?? "Unknown node"}: ${allocation.layers} layers`;
                  })
                  .join(" · ")}
              </p>
            )
          )}
          {running ? (
            <button className="button stop-button" onClick={() => void cancelBenchmark()}>
              Cancel benchmark
            </button>
          ) : (
            <button
              className="button primary"
              disabled={!modelId}
              onClick={() => void runBenchmark()}
            >
              Run benchmark
            </button>
          )}
          <button className="text-button" onClick={() => navigate("models")}>
            Benchmark a model
          </button>
        </div>
      </header>
      {error && (
        <div className="error-panel" role="alert">
          {error}
        </div>
      )}
      {runs.length === 0 ? (
        <div className="empty-state">
          <span>▶</span>
          <div>
            <h2>No benchmark runs</h2>
            <p>Choose a model to compare valid single-node and distributed placements.</p>
          </div>
        </div>
      ) : (
        <div className="benchmark-table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Model / topology</th>
                <th>Prompt</th>
                <th>Generation</th>
                <th>Duration</th>
                <th>Peak memory</th>
                <th>Run</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((benchmark) => (
                <tr key={benchmark.id} className={benchmark.recommended ? "recommended" : ""}>
                  <td>
                    <strong>{benchmark.modelName}</strong>
                    <span className="capitalize">
                      {benchmark.error ? `Failed · ${benchmark.error}` : ""}
                      {benchmark.topology}
                      {benchmark.gpuLayers?.length
                        ? ` · ${benchmark.gpuLayers.map((item) => item.layers).join("/")} GPU layers`
                        : ""}
                      {benchmark.recommended && <i>Recommended</i>}
                    </span>
                  </td>
                  <td>
                    <strong>{benchmark.promptTokensPerSecond.toFixed(1)}</strong>
                    <small>tok/s</small>
                  </td>
                  <td>
                    <strong>{benchmark.generationTokensPerSecond.toFixed(1)}</strong>
                    <small>tok/s</small>
                  </td>
                  <td>{benchmark.loadTimeSeconds.toFixed(1)} s</td>
                  <td>
                    {benchmark.memoryPeakGb > 0 ? `${benchmark.memoryPeakGb.toFixed(1)} GB` : "—"}
                  </td>
                  <td>{formatRunTime(benchmark.ranAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="benchmark-note">
        <span>i</span>
        <p>
          <strong>Why results differ</strong> Prompt processing and token generation stress the link
          differently. SharedLocalLLM recommends the fastest valid result; distribution is not
          assumed to be faster.
        </p>
      </div>
    </div>
  );
}
