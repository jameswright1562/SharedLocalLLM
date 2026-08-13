import type { GpuLayerAllocation, ModelRecord, NodeCapabilities, SplitEstimate } from "../types";
import { fitLabels, formatContext } from "../pages/pageFormat";

interface ModelInspectorProps {
  selected?: ModelRecord;
  nodeLookup: Map<string, string>;
  contextSize: number;
  setContextSize: (contextSize: number) => void;
  manualSplit: boolean;
  setManualSplit: (manual: boolean) => void;
  gpuNodes: NodeCapabilities[];
  gpuLayers: GpuLayerAllocation[];
  splitEstimate?: SplitEstimate;
  setGpuLayers: (layers: GpuLayerAllocation[]) => void;
  busy: boolean;
  splitInvalid: boolean;
  launch: () => void;
}

export function ModelInspector({
  selected,
  nodeLookup,
  contextSize,
  setContextSize,
  manualSplit,
  setManualSplit,
  gpuNodes,
  gpuLayers,
  splitEstimate,
  setGpuLayers,
  busy,
  splitInvalid,
  launch,
}: ModelInspectorProps) {
  if (!selected)
    return (
      <aside className="model-inspector" aria-label="Selected model details">
        <div className="inspector-empty">Select a model to inspect fit and launch settings.</div>
      </aside>
    );

  return (
    <aside className="model-inspector" aria-label="Selected model details">
      <p className="section-kicker">Selected model</p>
      <h2>{selected.name}</h2>
      <dl>
        <div>
          <dt>Fit</dt>
          <dd>
            <span className={`fit-badge fit-${selected.fit}`}>{fitLabels[selected.fit]}</span>
          </dd>
        </div>
        <div>
          <dt>Context</dt>
          <dd>{formatContext(selected.contextLength)} tokens</dd>
        </div>
        <div>
          <dt>Format</dt>
          <dd>{selected.quantization}</dd>
        </div>
        <div>
          <dt>Files</dt>
          <dd>
            {selected.shards} GGUF{selected.capability === "vision" ? " + projector" : ""}
          </dd>
        </div>
        <div>
          <dt>Location</dt>
          <dd>
            {selected.locations[0]
              ? (nodeLookup.get(selected.locations[0].nodeId) ?? "Unknown node")
              : "Unknown node"}
          </dd>
        </div>
      </dl>
      <FitExplanation model={selected} />
      <label className="field-label" htmlFor="context-select">
        Requested context
      </label>
      <select
        id="context-select"
        value={String(contextSize)}
        onChange={(event) => setContextSize(Number(event.target.value))}
      >
        <option value="4096">4,096 tokens</option>
        <option value="8192">8,192 tokens</option>
        {selected.contextLength >= 16384 && <option value="16384">16,384 tokens</option>}
      </select>
      {selected.layerCount ? (
        <GpuAllocation
          selected={selected}
          manualSplit={manualSplit}
          setManualSplit={setManualSplit}
          gpuNodes={gpuNodes}
          gpuLayers={gpuLayers}
          splitEstimate={splitEstimate}
          setGpuLayers={setGpuLayers}
        />
      ) : (
        <p className="metadata-note">
          Manual layer allocation is unavailable because this GGUF does not expose layer metadata.
          Automatic allocation remains available.
        </p>
      )}
      <button
        className="button primary full"
        disabled={busy || selected.fit === "does-not-fit" || splitInvalid}
        onClick={launch}
        aria-label={`Launch ${selected.name}`}
      >
        {busy ? "Starting cluster…" : `Launch ${selected.name}`}
      </button>
    </aside>
  );
}

function FitExplanation({ model }: { model: ModelRecord }) {
  const heading =
    model.fit === "single-node"
      ? "Fast local placement"
      : model.fit === "combined-gpu"
        ? "Both GPUs required"
        : model.fit === "gpu-ram"
          ? "System memory assists"
          : "Insufficient safe memory";
  const detail =
    model.fit === "single-node"
      ? "This model fits on at least one GPU. Benchmarks decide whether distribution is worthwhile."
      : model.fit === "combined-gpu"
        ? "The model will be layer-split across the private link."
        : model.fit === "gpu-ram"
          ? "Some layers will use coordinator RAM, which can reduce speed."
          : "Free memory or choose a smaller quantization.";
  return (
    <div className="fit-explanation">
      <strong>{heading}</strong>
      <p>{detail}</p>
    </div>
  );
}

function GpuAllocation({
  selected,
  manualSplit,
  setManualSplit,
  gpuNodes,
  gpuLayers,
  splitEstimate,
  setGpuLayers,
}: Pick<
  ModelInspectorProps,
  "manualSplit" | "setManualSplit" | "gpuNodes" | "gpuLayers" | "splitEstimate" | "setGpuLayers"
> & { selected: ModelRecord }) {
  return (
    <section className="gpu-allocation" aria-label="GPU allocation mode">
      <div className="segmented allocation-mode" role="group" aria-label="GPU allocation">
        <button
          className={!manualSplit ? "active" : ""}
          aria-pressed={!manualSplit}
          onClick={() => setManualSplit(false)}
        >
          Automatic allocation
        </button>
        <button
          className={manualSplit ? "active" : ""}
          aria-pressed={manualSplit}
          onClick={() => setManualSplit(true)}
        >
          Manual GPU split
        </button>
      </div>
      {manualSplit && (
        <div className="gpu-split-panel">
          <div className="gpu-split-heading">
            <div>
              <h3>GPU layer allocation</h3>
              <p>Choose how many transformer layers each computer loads.</p>
            </div>
            <strong className={splitEstimate ? "" : "invalid"}>
              {splitEstimate
                ? `${splitEstimate.gpuLayers} of ${splitEstimate.totalLayers} layers on GPU`
                : `Too many of ${selected.layerCount} layers selected`}
            </strong>
          </div>
          <div className="gpu-device-list">
            {gpuNodes.map((node) => {
              const allocation = gpuLayers.find((item) => item.nodeId === node.id);
              const estimate = splitEstimate?.devices.find((device) => device.nodeId === node.id);
              return (
                <div className="gpu-device-allocation" key={node.id}>
                  <label>
                    <span>{node.name}</span>
                    <input
                      type="number"
                      min="0"
                      max={selected.layerCount}
                      value={allocation?.layers ?? 0}
                      aria-label={`GPU layers on ${node.name}`}
                      onChange={(event) => {
                        const layers = Math.max(
                          0,
                          Math.min(
                            selected.layerCount ?? 0,
                            Number.parseInt(event.target.value || "0", 10),
                          ),
                        );
                        setGpuLayers(
                          gpuLayers.map((item) =>
                            item.nodeId === node.id ? { ...item, layers } : item,
                          ),
                        );
                      }}
                    />
                  </label>
                  <VramEstimate node={node} estimate={estimate} />
                </div>
              );
            })}
          </div>
          {splitEstimate && (
            <p className="split-summary">
              {splitEstimate.cpuLayers
                ? `${splitEstimate.cpuLayers} layers remain on CPU · about ${formatMib(splitEstimate.estimatedCpuRamMib)} model RAM`
                : "All model layers are assigned to GPUs"}
              {!splitEstimate.usesAttentionMetadata && " · KV cache uses a conservative fallback"}
            </p>
          )}
          <p className="estimate-note">
            Estimates include model weights, F16 KV cache, and a 512 MiB runtime allowance per
            active GPU. Layer counts are target proportions; llama.cpp may round placement at tensor
            boundaries.
          </p>
        </div>
      )}
    </section>
  );
}

function VramEstimate({
  node,
  estimate,
}: {
  node: NodeCapabilities;
  estimate?: SplitEstimate["devices"][number];
}) {
  return (
    <div className="vram-estimate">
      <span>Estimated VRAM</span>
      <strong>{estimate ? formatMib(estimate.estimatedVramMib) : "—"}</strong>
      <small>
        {formatMib(estimate?.availableVramMib ?? node.gpu.vramAvailableGb * 1024)} available
      </small>
      {estimate && (
        <i className={estimate.fits ? "fits" : "over"}>
          {estimate.fits ? "Fits current VRAM" : "Exceeds current VRAM"}
        </i>
      )}
    </div>
  );
}

function formatMib(value: number) {
  return value >= 1024 ? `${(value / 1024).toFixed(2)} GiB` : `${Math.ceil(value)} MiB`;
}
