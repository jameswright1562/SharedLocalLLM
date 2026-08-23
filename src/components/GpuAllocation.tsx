import type { ReactNode } from "react";
import {
  Checkbox,
  Group,
  Paper,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import type {
  GpuLayerAllocation,
  ModelRecord,
  NodeCapabilities,
  SplitEstimate,
} from "../types";
import { formatMib } from "./formatMib";

interface GpuAllocationProps {
  selected: ModelRecord;
  manualSplit: boolean;
  setManualSplit: (manual: boolean) => void;
  gpuNodes: NodeCapabilities[];
  gpuLayers: GpuLayerAllocation[];
  splitEstimate?: SplitEstimate;
  setGpuLayers: (layers: GpuLayerAllocation[]) => void;
  workerNode?: NodeCapabilities;
  includeRemoteCpu: boolean;
  setIncludeRemoteCpu: (value: boolean) => void;
}

export function GpuAllocation({
  selected,
  manualSplit,
  setManualSplit,
  gpuNodes,
  gpuLayers,
  splitEstimate,
  setGpuLayers,
  workerNode,
  includeRemoteCpu,
  setIncludeRemoteCpu,
}: GpuAllocationProps) {
  return (
    <Paper component="section" aria-label="GPU allocation mode" withBorder p="md" mt="md">
      <SegmentedControl
        fullWidth
        aria-label="GPU allocation"
        value={manualSplit ? "manual" : "auto"}
        onChange={(value) => setManualSplit(value === "manual")}
        data={[
          { value: "auto", label: "Automatic allocation" },
          { value: "manual", label: "Manual GPU split" },
        ]}
        mb="sm"
      />
      {manualSplit && (
        <Stack gap="sm">
          <Group justify="space-between" align="flex-start" wrap="nowrap">
            <div>
              <Title order={4}>GPU layer allocation</Title>
              <Text size="xs" c="dimmed">
                Choose how many transformer layers each computer loads.
              </Text>
            </div>
            <Text size="xs" fw={700} ta="right" c={splitEstimate ? undefined : "coral"}>
              {splitEstimate
                ? `${splitEstimate.gpuLayers} of ${splitEstimate.totalLayers} layers on GPU`
                : `Too many of ${selected.layerCount} layers selected`}
            </Text>
          </Group>
          <Stack gap="xs">
            {gpuNodes.map((node) => (
              <DeviceAllocation
                key={node.id}
                node={node}
                kind="gpu"
                name={node.name}
                selected={selected}
                allocation={gpuLayers.find((item) => item.nodeId === node.id)}
                estimate={splitEstimate?.devices.find(
                  (device) => device.nodeId === node.id && !device.kind,
                )}
                gpuLayers={gpuLayers}
                setGpuLayers={setGpuLayers}
              />
            ))}
          </Stack>
          {workerNode && (
            <Stack gap="xs">
              <Checkbox
                label={<Text size="sm">Offload layers to {workerNode.name}&apos;s CPU</Text>}
                checked={includeRemoteCpu}
                onChange={(event) => setIncludeRemoteCpu(event.currentTarget.checked)}
              />
              {includeRemoteCpu && (
                <DeviceAllocation
                  node={workerNode}
                  kind="cpu"
                  name={`${workerNode.name} CPU`}
                  selected={selected}
                  allocation={gpuLayers.find(
                    (item) => item.nodeId === workerNode.id && item.kind === "cpu",
                  )}
                  estimate={splitEstimate?.devices.find(
                    (device) => device.nodeId === workerNode.id && device.kind === "cpu",
                  )}
                  gpuLayers={gpuLayers}
                  setGpuLayers={setGpuLayers}
                />
              )}
            </Stack>
          )}
          {splitEstimate && (
            <Text size="xs" c="dimmed">
              {splitEstimate.cpuLayers
                ? `${splitEstimate.cpuLayers} layers remain on CPU · about ${formatMib(splitEstimate.estimatedCpuRamMib)} model RAM`
                : "All model layers are assigned to GPUs"}
              {!splitEstimate.usesAttentionMetadata &&
                " · KV cache uses a conservative fallback"}
            </Text>
          )}
          <Text size="xs" c="dimmed" lh={1.35}>
            Estimates include model weights, F16 KV cache, and a 512 MiB runtime allowance per
            active GPU. Layer counts are target proportions; llama.cpp may round placement at tensor
            boundaries.
          </Text>
        </Stack>
      )}
    </Paper>
  );
}

function DeviceAllocation({
  node,
  kind,
  name,
  selected,
  allocation,
  estimate,
  gpuLayers,
  setGpuLayers,
}: {
  node: NodeCapabilities;
  kind: "gpu" | "cpu";
  name: string;
  selected: ModelRecord;
  allocation?: GpuLayerAllocation;
  estimate?: SplitEstimate["devices"][number];
  gpuLayers: GpuLayerAllocation[];
  setGpuLayers: (layers: GpuLayerAllocation[]) => void;
}) {
  const maxLayers = selected.layerCount ?? 0;
  const id = `layer-allocation-${node.id}-${kind}`;

  function updateLayers(rawValue: string) {
    const layers = Math.max(
      0,
      Math.min(maxLayers, Number.parseInt(rawValue || "0", 10)),
    );
    if (kind === "gpu") {
      setGpuLayers(
        gpuLayers.map((item) =>
          item.nodeId === node.id && !item.kind ? { ...item, layers } : item,
        ),
      );
      return;
    }
    // CPU offload: trim any overflow from the worker GPU share first, then from
    // every other share, so total assigned layers never exceed the layer count.
    const withoutCpu = gpuLayers.filter(
      (item) => !(item.nodeId === node.id && item.kind === "cpu"),
    );
    let overflow = Math.max(
      0,
      withoutCpu.reduce((total, item) => total + item.layers, 0) + layers - maxLayers,
    );
    const trimWorkerFirst = (item: GpuLayerAllocation) => {
      if (overflow === 0 || item.nodeId !== node.id || item.kind) return item;
      const removed = Math.min(item.layers, overflow);
      overflow -= removed;
      return { ...item, layers: item.layers - removed };
    };
    const trimAny = (item: GpuLayerAllocation) => {
      if (overflow === 0) return item;
      const removed = Math.min(item.layers, overflow);
      overflow -= removed;
      return { ...item, layers: item.layers - removed };
    };
    setGpuLayers([
      ...withoutCpu.map(trimWorkerFirst).map(trimAny),
      { nodeId: node.id, layers, kind: "cpu" },
    ]);
  }

  return (
    <Group justify="space-between" gap="md" wrap="nowrap" align="center">
      <Text component="label" htmlFor={id} size="sm" style={{ flex: 1 }}>
        {name}
      </Text>
      <TextInput
        id={id}
        type="number"
        min={0}
        max={maxLayers}
        w={90}
        value={allocation?.layers ?? 0}
        aria-label={`${kind === "cpu" ? "CPU" : "GPU"} layers on ${node.name}`}
        onChange={(event) => updateLayers(event.target.value)}
      />
      <VramEstimate node={node} estimate={estimate} />
    </Group>
  );
}

function VramEstimate({
  node,
  estimate,
}: {
  node: NodeCapabilities;
  estimate?: SplitEstimate["devices"][number];
}): ReactNode {
  const cpu = estimate?.kind === "cpu";
  const memory = cpu ? "RAM" : "VRAM";
  const available =
    estimate?.availableVramMib ?? (cpu ? node.ramAvailableGb : node.gpu.vramAvailableGb) * 1024;
  return (
    <Stack gap={0} w={170} maw="42%">
      <Text size="xs" c="dimmed">
        Estimated {memory}:{" "}
        <b>{estimate ? formatMib(estimate.estimatedVramMib) : "—"}</b>
      </Text>
      <Text size="10px" c="dimmed">
        {formatMib(available)} available
      </Text>
      {estimate && (
        <Text size="10px" fw={600} c={estimate.fits ? "mint" : "coral"}>
          {estimate.fits ? `Fits current ${memory}` : `Exceeds current ${memory}`}
        </Text>
      )}
    </Stack>
  );
}
