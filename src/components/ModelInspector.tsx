import { useState } from "react";
import type {
  GpuLayerAllocation,
  ModelLoadOptions,
  ModelRecord,
  NodeCapabilities,
  SplitEstimate,
} from "../types";
import { fitLabels, formatContext } from "../pages/pageFormat";
import { AdvancedLoadOptions } from "./LoadOptions";

const MIN_CONTEXT = 4096;

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
  loadOptions: ModelLoadOptions;
  setLoadOptions: (options: ModelLoadOptions) => void;
  busy: boolean;
  splitInvalid: boolean;
  force: boolean;
  setForce: (force: boolean) => void;
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
  loadOptions,
  setLoadOptions,
  busy,
  splitInvalid,
  force,
  setForce,
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
      <ContextSizeControl
        contextSize={contextSize}
        setContextSize={setContextSize}
        maxContext={selected.contextLength}
      />
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
      <AdvancedLoadOptions options={loadOptions} setOptions={setLoadOptions} />
      <button
        className="button primary full"
        disabled={
          busy ||
          selected.remoteOnly ||
          ((selected.fit === "does-not-fit" || splitInvalid) && !force)
        }
        onClick={launch}
        aria-label={`Launch ${selected.name}`}
      >
        {busy ? "Starting cluster…" : `Launch ${selected.name}`}
      </button>
      {selected.remoteOnly ? (
        <p className="metadata-note">
          This GGUF is stored on the other computer. Launch it there, or copy the file locally.
        </p>
      ) : selected.fit === "does-not-fit" || splitInvalid ? (
        <label className="force-launch">
          <input
            type="checkbox"
            checked={force}
            onChange={(event) => setForce(event.target.checked)}
          />
          <span>Force launch — ignore the memory estimate</span>
        </label>
      ) : null}
      {force && (
        <p className="metadata-note">
          Forced launch disables the fit check. The model may load slowly, spill heavily, or fail to
          start if memory is genuinely insufficient.
        </p>
      )}
    </aside>
  );
}

function ContextSizeControl({
  contextSize,
  setContextSize,
  maxContext,
}: {
  contextSize: number;
  setContextSize: (value: number) => void;
  maxContext: number;
}) {
  const max = Math.max(MIN_CONTEXT, maxContext);
  const [draft, setDraft] = useState(String(contextSize));
  const [previousContextSize, setPreviousContextSize] = useState(contextSize);

  if (contextSize !== previousContextSize) {
    setPreviousContextSize(contextSize);
    setDraft(String(contextSize));
  }

  function commit(value: number) {
    const clamped = Math.max(MIN_CONTEXT, Math.min(max, Math.round(value)));
    setContextSize(clamped);
    setDraft(String(clamped));
  }

  return (
    <section className="context-control">
      <label className="field-label" htmlFor="context-size-input">
        Requested context
      </label>
      <input
        type="range"
        id="context-size-slider"
        min={MIN_CONTEXT}
        max={max}
        step={1024}
        value={Math.max(MIN_CONTEXT, Math.min(max, contextSize))}
        onChange={(event) => commit(Number(event.target.value))}
        aria-label="Requested context slider"
      />
      <div className="context-value-row">
        <input
          type="number"
          id="context-size-input"
          min={MIN_CONTEXT}
          max={max}
          value={draft}
          onChange={(event) => {
            const value = Number(event.target.value);
            if (event.target.value === "") {
              setDraft("");
            } else if (Number.isFinite(value)) {
              setDraft(event.target.value);
            }
          }}
          onBlur={(event) => {
            const value = Number(event.target.value);
            if (event.target.value === "") {
              setDraft(String(contextSize));
            } else if (Number.isFinite(value)) {
              commit(value);
            }
          }}
        />
        <span>{contextSize.toLocaleString()} tokens</span>
      </div>
    </section>
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
