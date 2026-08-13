import type { ModelRecord } from "../types";
import { fitLabels, formatBytes } from "../pages/pageFormat";

export type CapabilityFilter = "all" | "text" | "vision" | "split";

interface ModelCatalogueProps {
  models: ModelRecord[];
  visibleModels: ModelRecord[];
  selectedId: string;
  select: (id: string) => void;
  addFolder: () => void;
}

export function ModelCatalogue({
  models,
  visibleModels,
  selectedId,
  select,
  addFolder,
}: ModelCatalogueProps) {
  return (
    <div className="model-list" data-testid="model-list">
      {visibleModels.map((model) => (
        <button
          className={`model-row ${selectedId === model.id ? "selected" : ""}`}
          key={model.id}
          onClick={() => select(model.id)}
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
            <button className="button secondary" onClick={addFolder}>
              Add folder
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
