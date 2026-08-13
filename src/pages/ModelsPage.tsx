import { useMemo, useState } from "react";
import { describeAppError } from "../services/errors";
import type { PageProps } from "../types";
import { fitLabels, formatBytes, formatContext } from "./pageFormat";

type CapabilityFilter = "all" | "text" | "vision" | "split";

export function ModelsPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<CapabilityFilter>("all");
  const [selectedId, setSelectedId] = useState(snapshot.models[0]?.id ?? "");
  const [discoveredModels, setDiscoveredModels] = useState<typeof snapshot.models | null>(null);
  const [busy, setBusy] = useState<"refresh" | "add" | "launch" | "">("");
  const [message, setMessage] = useState("");
  const [contextByModel, setContextByModel] = useState<Record<string, number>>({});
  const nodeLookup = new Map(snapshot.nodes.map((node) => [node.id, node.name]));

  const models = discoveredModels ?? snapshot.models;

  const visibleModels = useMemo(
    () =>
      models.filter((model) => {
        const matchesQuery = `${model.name} ${model.architecture} ${model.quantization}`
          .toLowerCase()
          .includes(query.toLowerCase());
        const matchesFilter =
          filter === "all" || (filter === "split" ? model.shards > 1 : model.capability === filter);
        return matchesQuery && matchesFilter;
      }),
    [filter, models, query],
  );
  const selected = visibleModels.find((model) => model.id === selectedId) ?? visibleModels[0];
  const contextSize = selected
    ? (contextByModel[selected.id] ?? Math.min(selected.contextLength, 8192))
    : 8192;

  async function refreshModels() {
    setBusy("refresh");
    setMessage("");
    try {
      setDiscoveredModels(await service.discoverModels());
      setMessage("Model index refreshed.");
    } catch (reason) {
      setMessage(describeAppError(reason, "Model discovery failed."));
    } finally {
      setBusy("");
    }
  }

  async function addFolder() {
    setBusy("add");
    setMessage("");
    try {
      const directory = await service.addModelDirectory();
      if (!directory) {
        setMessage("No folder selected.");
        return;
      }
      await refreshSnapshot();
      setMessage("Model folder added.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The folder could not be added."));
    } finally {
      setBusy("");
    }
  }

  async function launch() {
    if (!selected || selected.fit === "does-not-fit") return;
    setBusy("launch");
    setMessage("");
    try {
      await service.startCluster(selected.id, contextSize);
      setMessage(`${selected.name} is loading on the recommended topology.`);
      await refreshSnapshot();
    } catch (reason) {
      setMessage(describeAppError(reason, "The model could not be launched."));
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="page models-page">
      <header className="page-header split-header">
        <div>
          <p className="section-kicker">Catalogue</p>
          <h1>Model library</h1>
          <p>GGUF models indexed across both computers. Source files stay where they are.</p>
        </div>
        <div className="button-row flush">
          <button
            className="button secondary"
            disabled={!!busy}
            onClick={() => void refreshModels()}
          >
            {busy === "refresh" ? "Indexing…" : "Refresh"}
          </button>
          <button className="button primary" disabled={!!busy} onClick={() => void addFolder()}>
            Add folder
          </button>
        </div>
      </header>
      <div className="model-toolbar">
        <label className="search-field">
          <span aria-hidden="true">⌕</span>
          <input
            type="search"
            aria-label="Search models"
            placeholder="Search name, architecture, quantization…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <div className="segmented" role="group" aria-label="Filter model type">
          {(["all", "text", "vision", "split"] as CapabilityFilter[]).map((value) => (
            <button
              key={value}
              className={filter === value ? "active" : ""}
              onClick={() => setFilter(value)}
              aria-pressed={filter === value}
            >
              {value}
            </button>
          ))}
        </div>
      </div>
      <div className="model-layout">
        <div className="model-list" data-testid="model-list">
          {visibleModels.map((model) => (
            <button
              className={`model-row ${selectedId === model.id ? "selected" : ""}`}
              key={model.id}
              onClick={() => setSelectedId(model.id)}
              aria-pressed={selectedId === model.id}
            >
              <span className={`model-kind kind-${model.capability}`}>
                {model.capability === "vision" ? "◉" : "T"}
              </span>
              <span className="model-main">
                <strong>{model.name}</strong>
                <small>
                  {model.architecture} · {model.quantization} · {formatBytes(model.sizeBytes)}
                </small>
              </span>
              <span className="model-tags">
                <i>{model.capability}</i>
                {model.shards > 1 && <i>{model.shards} shards</i>}
                <i>{model.locations[0]?.source === "lm-studio" ? "LM Studio" : "Custom"}</i>
              </span>
              <span className={`fit-badge fit-${model.fit}`}>{fitLabels[model.fit]}</span>
              <span className="row-arrow" aria-hidden="true">
                ›
              </span>
            </button>
          ))}
          {visibleModels.length === 0 && (
            <div className="empty-state model-empty">
              <span>GG</span>
              <div>
                <h2>{models.length ? "No models match" : "No models indexed"}</h2>
                <p>
                  {models.length
                    ? "Clear the search or choose another filter."
                    : "Add an LM Studio or custom folder containing GGUF files."}
                </p>
                <button className="button secondary" onClick={() => void addFolder()}>
                  Add folder
                </button>
              </div>
            </div>
          )}
        </div>
        <aside className="model-inspector" aria-label="Selected model details">
          {selected ? (
            <>
              <p className="section-kicker">Selected model</p>
              <h2>{selected.name}</h2>
              <dl>
                <div>
                  <dt>Fit</dt>
                  <dd>
                    <span className={`fit-badge fit-${selected.fit}`}>
                      {fitLabels[selected.fit]}
                    </span>
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
              <div className="fit-explanation">
                <strong>
                  {selected.fit === "single-node"
                    ? "Fast local placement"
                    : selected.fit === "combined-gpu"
                      ? "Both GPUs required"
                      : selected.fit === "gpu-ram"
                        ? "System memory assists"
                        : "Insufficient safe memory"}
                </strong>
                <p>
                  {selected.fit === "single-node"
                    ? "This model fits on at least one GPU. Benchmarks decide whether distribution is worthwhile."
                    : selected.fit === "combined-gpu"
                      ? "The model will be layer-split across the private link."
                      : selected.fit === "gpu-ram"
                        ? "Some layers will use coordinator RAM, which can reduce speed."
                        : "Free memory or choose a smaller quantization."}
                </p>
              </div>
              <label className="field-label" htmlFor="context-select">
                Requested context
              </label>
              <select
                id="context-select"
                value={String(contextSize)}
                onChange={(event) =>
                  setContextByModel({
                    ...contextByModel,
                    [selected.id]: Number(event.target.value),
                  })
                }
              >
                <option value="4096">4,096 tokens</option>
                <option value="8192">8,192 tokens</option>
                {selected.contextLength >= 16384 && <option value="16384">16,384 tokens</option>}
              </select>
              <button
                className="button primary full"
                disabled={busy === "launch" || selected.fit === "does-not-fit"}
                onClick={() => void launch()}
                aria-label={`Launch ${selected.name}`}
              >
                {busy === "launch" ? "Starting cluster…" : `Launch ${selected.name}`}
              </button>
            </>
          ) : (
            <div className="inspector-empty">
              Select a model to inspect fit and launch settings.
            </div>
          )}
        </aside>
      </div>
      {message && (
        <div className="toast-message" role="status">
          {message}
        </div>
      )}
    </div>
  );
}
