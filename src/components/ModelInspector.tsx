import { useState, type ReactNode } from "react";
import {
  Badge,
  Button,
  Checkbox,
  DataList,
  Group,
  Paper,
  Slider,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import type {
  GpuLayerAllocation,
  ModelLoadOptions,
  ModelRecord,
  NodeCapabilities,
  SplitEstimate,
} from "../types";
import { fitLabels, formatContext } from "../pages/pageFormat";
import { AdvancedLoadOptions } from "./LoadOptions";
import { GpuAllocation } from "./GpuAllocation";

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
  workerNode?: NodeCapabilities;
  includeRemoteCpu: boolean;
  setIncludeRemoteCpu: (value: boolean) => void;
  loadOptions: ModelLoadOptions;
  setLoadOptions: (options: ModelLoadOptions) => void;
  busy: boolean;
  splitInvalid: boolean;
  force: boolean;
  setForce: (force: boolean) => void;
  launch: () => void;
  autotuneSection?: ReactNode;
}

const fitBadgeColors = {
  "single-node": "mint",
  "combined-gpu": "cyan",
  "gpu-ram": "amber",
  "does-not-fit": "coral",
} as const;

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
  workerNode,
  includeRemoteCpu,
  setIncludeRemoteCpu,
  loadOptions,
  setLoadOptions,
  busy,
  splitInvalid,
  force,
  setForce,
  launch,
  autotuneSection,
}: ModelInspectorProps) {
  if (!selected)
    return (
      <Paper aria-label="Selected model details" p="md">
        <Text c="dimmed">Select a model to inspect fit and launch settings.</Text>
      </Paper>
    );

  return (
    <Stack gap="sm" aria-label="Selected model details">
      <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
        Selected model
      </Text>
      <Title order={2}>{selected.name}</Title>
      <DataList
        size="sm"
        styles={{ itemLabel: { width: 90, color: "var(--mantine-color-dimmed)" } }}
      >
        <DataList.Item>
          <DataList.ItemLabel>Fit</DataList.ItemLabel>
          <DataList.ItemValue>
            <Badge
              color={fitBadgeColors[selected.fit]}
              variant="light"
              className={`fit-${selected.fit}`}
            >
              {fitLabels[selected.fit]}
            </Badge>
          </DataList.ItemValue>
        </DataList.Item>
        <DataList.Item>
          <DataList.ItemLabel>Context</DataList.ItemLabel>
          <DataList.ItemValue>{formatContext(selected.contextLength)} tokens</DataList.ItemValue>
        </DataList.Item>
        <DataList.Item>
          <DataList.ItemLabel>Format</DataList.ItemLabel>
          <DataList.ItemValue>{selected.quantization}</DataList.ItemValue>
        </DataList.Item>
        <DataList.Item>
          <DataList.ItemLabel>Files</DataList.ItemLabel>
          <DataList.ItemValue>
            {selected.shards} GGUF{selected.capability === "vision" ? " + projector" : ""}
          </DataList.ItemValue>
        </DataList.Item>
        <DataList.Item>
          <DataList.ItemLabel>Location</DataList.ItemLabel>
          <DataList.ItemValue>
            {selected.locations[0]
              ? (nodeLookup.get(selected.locations[0].nodeId) ?? "Unknown node")
              : "Unknown node"}
          </DataList.ItemValue>
        </DataList.Item>
      </DataList>
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
          workerNode={workerNode}
          includeRemoteCpu={includeRemoteCpu}
          setIncludeRemoteCpu={setIncludeRemoteCpu}
        />
      ) : (
        <Note>
          Manual layer allocation is unavailable because this GGUF does not expose layer metadata.
          Automatic allocation remains available.
        </Note>
      )}
      <AdvancedLoadOptions options={loadOptions} setOptions={setLoadOptions} />
      {autotuneSection}
      <Button
        fullWidth
        disabled={busy || ((selected.fit === "does-not-fit" || splitInvalid) && !force)}
        onClick={launch}
        aria-label={`Launch ${selected.name}`}
      >
        {busy ? "Starting cluster…" : `Launch ${selected.name}`}
      </Button>
      {selected.remoteOnly && (
        <Note>
          This GGUF is stored on the other computer. Launching here asks that computer to coordinate
          the model while this app remains the controller.
        </Note>
      )}
      {(selected.fit === "does-not-fit" || splitInvalid) && (
        <Checkbox
          label={<Text size="sm">Force launch — ignore the memory estimate</Text>}
          checked={force}
          onChange={(event) => setForce(event.currentTarget.checked)}
        />
      )}
      {force && (
        <Note>
          Forced launch disables the fit check. The model may load slowly, spill heavily, or fail to
          start if memory is genuinely insufficient.
        </Note>
      )}
    </Stack>
  );
}

function Note({ children }: { children: string }) {
  return (
    <Text size="xs" c="dimmed" lh={1.4}>
      {children}
    </Text>
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
    <Paper p="sm" bg="dark.8">
      <Text size="sm" fw={600}>
        {heading}
      </Text>
      <Text size="sm" c="dimmed">
        {detail}
      </Text>
    </Paper>
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
    <Paper component="section" withBorder p="md">
      <Text component="label" htmlFor="context-size-input" size="sm" fw={500}>
        Requested context
      </Text>
      <Slider
        aria-label="Requested context slider"
        min={MIN_CONTEXT}
        max={max}
        step={1024}
        value={Math.max(MIN_CONTEXT, Math.min(max, contextSize))}
        onChange={(event) => commit(event)}
        color="cyan"
        my="sm"
        label={(value) => value.toLocaleString()}
      />
      <Group justify="space-between" align="center" wrap="nowrap">
        <TextInput
          id="context-size-input"
          type="number"
          min={MIN_CONTEXT}
          max={max}
          w={130}
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
        <Group gap={4}>
          <Text size="sm" c="dimmed">
            {contextSize.toLocaleString()} tokens
          </Text>
        </Group>
      </Group>
    </Paper>
  );
}
