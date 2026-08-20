import { useMemo, useState } from "react";
import { ModelCatalogue, type CapabilityFilter } from "../components/ModelCatalogue";
import { ModelInspector } from "../components/ModelInspector";
import { Modal } from "@mantine/core";
import { describeAppError } from "../services/errors";
import { DEFAULT_LOAD_OPTIONS } from "../services/loadOptions";
import { distributeLayersByVram, estimateModelSplitLocally } from "../services/splitEstimate";
import type { GpuLayerAllocation, ModelLoadOptions, PageProps } from "../types";

export function ModelsPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<CapabilityFilter>("all");
  const [selectedId, setSelectedId] = useState(snapshot.models[0]?.id ?? "");
  const [discoveredModels, setDiscoveredModels] = useState<typeof snapshot.models | null>(null);
  const [busy, setBusy] = useState<"refresh" | "add" | "launch" | "">("");
  const [message, setMessage] = useState("");
  const [contextByModel, setContextByModel] = useState<Record<string, number>>({});
  const [manualByModel, setManualByModel] = useState<Record<string, boolean>>({});
  const [layersByModel, setLayersByModel] = useState<Record<string, GpuLayerAllocation[]>>({});
  const [remoteCpuByModel, setRemoteCpuByModel] = useState<Record<string, boolean>>({});
  const [forceByModel, setForceByModel] = useState<Record<string, boolean>>({});
  const [optionsByModel, setOptionsByModel] = useState<Record<string, ModelLoadOptions>>({});
  const [inspectorOpened, setInspectorOpened] = useState(false);
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
  const gpuNodes = useMemo(
    () => snapshot.nodes.filter((node) => node.online && node.gpu.vramTotalGb > 0),
    [snapshot.nodes],
  );
  const workerNode = snapshot.nodes.find((node) => node.role === "worker");
  const manualSplit = selected ? Boolean(manualByModel[selected.id]) : false;
  const includeRemoteCpu = selected ? Boolean(remoteCpuByModel[selected.id]) : false;
  const force = selected ? Boolean(forceByModel[selected.id]) : false;
  const loadOptions = selected
    ? (optionsByModel[selected.id] ?? DEFAULT_LOAD_OPTIONS)
    : DEFAULT_LOAD_OPTIONS;
  const gpuLayers = useMemo(
    () =>
      selected
        ? (layersByModel[selected.id] ?? distributeLayersByVram(selected.layerCount ?? 0, gpuNodes))
        : [],
    [gpuNodes, layersByModel, selected],
  );
  const splitEstimate = useMemo(() => {
    if (!selected?.layerCount || !manualSplit) return undefined;
    try {
      return estimateModelSplitLocally(selected, gpuNodes, { contextSize, gpuLayers });
    } catch {
      return undefined;
    }
  }, [contextSize, gpuLayers, gpuNodes, manualSplit, selected]);
  const splitInvalid =
    manualSplit &&
    (!splitEstimate ||
      splitEstimate.gpuLayers === 0 ||
      splitEstimate.devices.some((device) => !device.fits));

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
      setDiscoveredModels(null);
      await refreshSnapshot();
      setMessage("Model folder added.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The folder could not be added."));
    } finally {
      setBusy("");
    }
  }

  async function launch() {
    if (!selected) return;
    const forceLaunch = Boolean(forceByModel[selected.id]);
    if (!forceLaunch && (selected.fit === "does-not-fit" || splitInvalid)) return;
    setBusy("launch");
    setMessage("");
    try {
      await service.startCluster(selected.id, {
        contextSize,
        gpuLayers: selected.layerCount ? gpuLayers : [],
        includeRemoteCpu,
        force: forceLaunch,
        flashAttention: loadOptions.flashAttention,
        useMmap: loadOptions.useMmap,
        useMlock: loadOptions.useMlock,
        cpuThreads: loadOptions.cpuThreads,
        batchSize: loadOptions.batchSize,
      });
      setMessage(
        `${selected.name} is loading with ${manualSplit ? "the selected GPU layer split" : "automatic allocation"}${forceLaunch ? " (forced)" : ""}.`,
      );
      await refreshSnapshot();
    } catch (reason) {
      setMessage(describeAppError(reason, "The model could not be launched."));
    } finally {
      setBusy("");
    }
  }

  function setManualSplit(manual: boolean) {
    if (!selected) return;
    setManualByModel({ ...manualByModel, [selected.id]: manual });
    if (manual && !layersByModel[selected.id]) {
      setLayersByModel({
        ...layersByModel,
        [selected.id]: distributeLayersByVram(selected.layerCount ?? 0, gpuNodes),
      });
    }
  }

  function setIncludeRemoteCpu(value: boolean) {
    if (!selected || !workerNode) return;
    setRemoteCpuByModel({ ...remoteCpuByModel, [selected.id]: value });
    const current =
      layersByModel[selected.id] ?? distributeLayersByVram(selected.layerCount ?? 0, gpuNodes);
    const withoutCpu = current.filter(
      (item) => !(item.nodeId === workerNode.id && item.kind === "cpu"),
    );
    setLayersByModel({
      ...layersByModel,
      [selected.id]: value
        ? [...withoutCpu, { nodeId: workerNode.id, layers: 0, kind: "cpu" }]
        : withoutCpu,
    });
  }

  function setForce(next: boolean) {
    if (!selected) return;
    setForceByModel({ ...forceByModel, [selected.id]: next });
  }

  return (
    <div className="page models-page">
      <header className="page-header split-header">
        <div>
          <p className="section-kicker">Catalogue</p>
          <h1>Model library</h1>
          <p>
            GGUF models on this computer, plus names reported by a paired peer. Launch requires a
            local file. Source files stay where they are.
          </p>
        </div>
        <div className="button-row flush">
          <button
            className="button secondary"
            disabled={!!busy}
            onClick={() => void refreshModels()}
          >
            {busy === "refresh" ? "Indexing…" : "Refresh"}
          </button>
          {(snapshot.cluster.status === "running" || snapshot.cluster.status === "loading") && (
            <button
              className="button stop-button"
              disabled={!!busy}
              onClick={() => void service.stopCluster().then(() => refreshSnapshot())}
            >
              Stop cluster
            </button>
          )}
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
        <ModelCatalogue
          models={models}
          visibleModels={visibleModels}
          selectedId={selectedId}
          select={(id) => {
            setSelectedId(id);
            setInspectorOpened(true);
          }}
          addFolder={() => void addFolder()}
        />
        <Modal
          opened={inspectorOpened}
          onClose={() => setInspectorOpened(false)}
          title="Model inspector"
          size="xl"
          centered
        >
          <ModelInspector
            selected={selected}
            nodeLookup={nodeLookup}
            contextSize={contextSize}
            setContextSize={(value) =>
              selected && setContextByModel({ ...contextByModel, [selected.id]: value })
            }
            manualSplit={manualSplit}
            setManualSplit={setManualSplit}
            gpuNodes={gpuNodes}
            gpuLayers={gpuLayers}
            splitEstimate={splitEstimate}
            setGpuLayers={(layers) =>
              selected && setLayersByModel({ ...layersByModel, [selected.id]: layers })
            }
            workerNode={workerNode}
            includeRemoteCpu={includeRemoteCpu}
            setIncludeRemoteCpu={setIncludeRemoteCpu}
            loadOptions={loadOptions}
            setLoadOptions={(options) =>
              selected && setOptionsByModel({ ...optionsByModel, [selected.id]: options })
            }
            busy={busy === "launch"}
            splitInvalid={splitInvalid}
            force={force}
            setForce={setForce}
            launch={() => void launch()}
          />
        </Modal>
      </div>
      {message && (
        <div className="toast-message" role="status">
          {message}
        </div>
      )}
    </div>
  );
}
