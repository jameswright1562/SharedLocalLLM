import { useMemo, useState } from "react";
import {
  Box,
  Button,
  Flex,
  Group,
  Modal,
  SegmentedControl,
  Stack,
  TextInput,
  Title,
  Text,
} from "@mantine/core";
import { IconSearch } from "@tabler/icons-react";

import { ModelCatalogue, type CapabilityFilter } from "../components/ModelCatalogue";
import { AutotunePanel, TunedBadge } from "../components/AutotunePanel";
import { ModelInspector } from "../components/ModelInspector";
import { StatusBanner } from "../components/StatusBanner";
import { modelFolder } from "../helpers/helpers";
import { formatBytes } from "./pageFormat";
import { describeAppError } from "../services/errors";
import { DEFAULT_LOAD_OPTIONS } from "../services/loadOptions";
import {
  savedContextSizes,
  savedForceLaunches,
  savedLayerSplits,
  savedManualSplits,
  savedOptionValues,
  savedRemoteCpuFlags,
} from "../services/savedLoadConfigs";
import {
  distributeLayersByVram,
  estimateModelSplitLocally,
  fitLayersByVram,
} from "../services/splitEstimate";
import type { ModelLoadConfig, ModelRecord, PageProps } from "../types";

const capabilityFilters: Array<{ value: CapabilityFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "text", label: "Text" },
  { value: "vision", label: "Vision" },
  { value: "split", label: "Split" },
];

const busyClusterStates = ["loading", "running", "stopping"];

export function ModelsPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<CapabilityFilter>("all");
  const [selectedId, setSelectedId] = useState(snapshot.models[0]?.id ?? "");
  const [discoveredModels, setDiscoveredModels] = useState<typeof snapshot.models | null>(null);
  const [busy, setBusy] = useState<"refresh" | "add" | "launch" | "delete" | "">("");
  const [message, setMessage] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ModelRecord | null>(null);
  const savedConfigs = snapshot.modelLoadConfigs ?? {};
  const [contextByModel, setContextByModel] = useState(() => savedContextSizes(savedConfigs));
  const [manualByModel, setManualByModel] = useState(() => savedManualSplits(savedConfigs));
  const [layersByModel, setLayersByModel] = useState(() => savedLayerSplits(savedConfigs));
  const [remoteCpuByModel, setRemoteCpuByModel] = useState(() => savedRemoteCpuFlags(savedConfigs));
  const [forceByModel, setForceByModel] = useState(() => savedForceLaunches(savedConfigs));
  const [optionsByModel, setOptionsByModel] = useState(() => savedOptionValues(savedConfigs));
  const [appliedConfigsByModel, setAppliedConfigsByModel] = useState<
    Record<string, ModelLoadConfig>
  >({});
  const [inspectorOpened, setInspectorOpened] = useState(false);
  const [tuneOpened, setTuneOpened] = useState(false);
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
  const activeNodes = useMemo(() => snapshot.nodes.filter((node) => node.online), [snapshot.nodes]);
  const workerNode = snapshot.nodes.find((node) => node.role === "worker");
  const manualSplit = selected ? Boolean(manualByModel[selected.id]) : false;
  const includeRemoteCpu = selected ? Boolean(remoteCpuByModel[selected.id]) : false;
  const force = selected ? Boolean(forceByModel[selected.id]) : false;
  const loadOptions = selected
    ? (optionsByModel[selected.id] ?? DEFAULT_LOAD_OPTIONS)
    : DEFAULT_LOAD_OPTIONS;
  const gpuLayers = useMemo(() => {
    if (!selected) return [];
    if (!manualSplit) return fitLayersByVram(selected, gpuNodes);
    return layersByModel[selected.id] ?? distributeLayersByVram(selected.layerCount ?? 0, gpuNodes);
  }, [gpuNodes, layersByModel, manualSplit, selected]);
  const splitEstimate = useMemo(() => {
    if (!selected?.layerCount || !manualSplit) return undefined;
    try {
      return estimateModelSplitLocally(selected, activeNodes, { contextSize, gpuLayers });
    } catch {
      return undefined;
    }
  }, [activeNodes, contextSize, gpuLayers, manualSplit, selected]);
  const splitInvalid =
    manualSplit &&
    (!splitEstimate ||
      splitEstimate.devices.every((device) => device.layers === 0) ||
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

  function requestDelete(model: ModelRecord) {
    if (busyClusterStates.includes(snapshot.cluster.status)) {
      setMessage("Stop the running cluster before deleting model files.");
      return;
    }
    setDeleteTarget(model);
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    const folder = modelFolder(deleteTarget.locations[0]?.path);
    if (!folder) return;
    setBusy("delete");
    setMessage("");
    try {
      await service.deleteModelFolder(folder);
      setDiscoveredModels(null);
      setDeleteTarget(null);
      await refreshSnapshot();
      setMessage(`Deleted ${deleteTarget.name} from disk.`);
    } catch (reason) {
      setMessage(describeAppError(reason, "The model folder could not be deleted."));
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
      const appliedConfig = appliedConfigsByModel[selected.id] ?? savedConfigs[selected.id];
      await service.startCluster(selected.id, {
        ...appliedConfig,
        contextSize,
        gpuLayers: selected.layerCount ? gpuLayers : [],
        includeRemoteCpu,
        force: forceLaunch,
        flashAttention: loadOptions.flashAttention,
        useMmap: loadOptions.useMmap,
        useMlock: loadOptions.useMlock,
        cpuThreads: loadOptions.cpuThreads,
        batchSize: loadOptions.batchSize,
        kvCacheK: loadOptions.kvCacheK || undefined,
        kvCacheV: loadOptions.kvCacheV || undefined,
        kvUnified: loadOptions.kvUnified ?? false,
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

  function applyTuneToEditor(config: ModelLoadConfig) {
    if (!selected) return;
    const modelId = selected.id;
    setAppliedConfigsByModel((current) => ({ ...current, [modelId]: config }));
    setContextByModel((current) => ({ ...current, [modelId]: config.contextSize }));
    setManualByModel((current) => ({
      ...current,
      [modelId]: config.gpuLayers.length > 0,
    }));
    setLayersByModel((current) => ({ ...current, [modelId]: config.gpuLayers }));
    setRemoteCpuByModel((current) => ({
      ...current,
      [modelId]: Boolean(config.includeRemoteCpu),
    }));
    setForceByModel((current) => ({ ...current, [modelId]: Boolean(config.force) }));
    setOptionsByModel((current) => ({
      ...current,
      [modelId]: {
        flashAttention: Boolean(config.flashAttention),
        useMmap: config.useMmap ?? DEFAULT_LOAD_OPTIONS.useMmap,
        useMlock: Boolean(config.useMlock),
        cpuThreads: config.cpuThreads ?? DEFAULT_LOAD_OPTIONS.cpuThreads,
        batchSize: config.batchSize ?? DEFAULT_LOAD_OPTIONS.batchSize,
        kvCacheK: config.kvCacheK ?? "",
        kvCacheV: config.kvCacheV ?? "",
        kvUnified: Boolean(config.kvUnified),
      },
    }));
  }

  return (
    <Box>
      <Flex justify="space-between" align="flex-start" gap="md" wrap="wrap" mb="lg">
        <Box>
          <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
            Catalogue
          </Text>
          <Title order={1}>Model library</Title>
          <Text c="dimmed" maw={560}>
            GGUF models on this computer, plus names reported by a paired peer. Launch requires a
            local file. Right-click a local model to open its folder or delete it from disk.
          </Text>
        </Box>
        <Group gap="sm">
          <Button variant="default" disabled={!!busy} onClick={() => void refreshModels()}>
            {busy === "refresh" ? "Indexing…" : "Refresh"}
          </Button>
          {(snapshot.cluster.status === "running" || snapshot.cluster.status === "loading") && (
            <Button
              color="coral"
              variant="light"
              disabled={!!busy}
              onClick={() => void service.stopCluster().then(() => refreshSnapshot())}
            >
              Stop cluster
            </Button>
          )}
          <Button disabled={!!busy} onClick={() => void addFolder()}>
            Add folder
          </Button>
        </Group>
      </Flex>

      <Flex gap="md" wrap="wrap" mb="md">
        <TextInput
          type="search"
          aria-label="Search models"
          placeholder="Search name, architecture, quantization…"
          leftSection={<IconSearch size={16} />}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          style={{ flex: 1, minWidth: 220 }}
        />
        <SegmentedControl
          aria-label="Filter model type"
          value={filter}
          onChange={(value) => setFilter(value as CapabilityFilter)}
          data={capabilityFilters}
        />
      </Flex>

      <ModelCatalogue
        models={models}
        visibleModels={visibleModels}
        selectedId={selectedId}
        select={(id) => {
          setSelectedId(id);
          setInspectorOpened(true);
        }}
        addFolder={() => void addFolder()}
        onDelete={requestDelete}
      />

      <Modal
        opened={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        title="Delete model folder"
        centered
      >
        <Stack gap="sm">
          <Text>
            Permanently delete <strong>{deleteTarget?.name}</strong> and its folder from this
            computer? This removes {formatBytes(deleteTarget?.sizeBytes ?? 0)} of model files and
            cannot be undone.
          </Text>
          <Text size="sm" c="dimmed" ff="monospace">
            {modelFolder(deleteTarget?.locations[0]?.path)}
          </Text>
          <Group justify="flex-end" gap="sm">
            <Button
              variant="default"
              disabled={busy === "delete"}
              onClick={() => setDeleteTarget(null)}
            >
              Cancel
            </Button>
            <Button color="red" loading={busy === "delete"} onClick={() => void confirmDelete()}>
              Delete folder
            </Button>
          </Group>
        </Stack>
      </Modal>

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
          autotuneSection={
            selected && !selected.remoteOnly ? (
              <Group gap="sm" align="center">
                <Button
                  variant="light"
                  color="cyan"
                  onClick={() => setTuneOpened(true)}
                  aria-label={`Auto-tune ${selected.name}`}
                >
                  Auto-tune settings…
                </Button>
                <TunedBadge tune={snapshot.modelTunes?.[selected.id]} />
              </Group>
            ) : undefined
          }
        />
      </Modal>

      <Modal
        opened={tuneOpened}
        onClose={() => setTuneOpened(false)}
        title={`Auto-tune ${selected?.name ?? ""}`.trim()}
        size="lg"
        centered
      >
        <AutotunePanel
          model={selected && !selected.remoteOnly ? selected : undefined}
          tune={selected ? snapshot.modelTunes?.[selected.id] : undefined}
          service={service}
          onMessage={setMessage}
          onRefresh={refreshSnapshot}
          onApplied={applyTuneToEditor}
        />
      </Modal>

      {message && <StatusBanner message={message} />}
    </Box>
  );
}
