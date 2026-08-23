import type { ModelLoadConfig, ModelRecord, NodeCapabilities, SplitEstimate } from "../types";

const MIB = 1_048_576;
const GPU_RUNTIME_ALLOWANCE_MIB = 512;

export function estimateModelSplitLocally(
  model: ModelRecord,
  nodes: NodeCapabilities[],
  loadConfig: ModelLoadConfig,
): SplitEstimate {
  const totalLayers = model.layerCount;
  if (!totalLayers) throw new Error("This model does not report a usable layer count.");

  const gpuLayers = loadConfig.gpuLayers
    .filter((item) => item.kind !== "cpu")
    .reduce((total, item) => total + item.layers, 0);
  const assignedLayers = loadConfig.gpuLayers.reduce((total, item) => total + item.layers, 0);
  if (assignedLayers > totalLayers)
    throw new Error(
      `The split selects ${assignedLayers} layers, but the model has ${totalLayers}.`,
    );

  const modelContext = Math.max(512, model.contextLength ?? 4096);
  const contextSize = Math.max(512, Math.min(loadConfig.contextSize, modelContext));
  const modelMib = Math.ceil(model.sizeBytes / MIB);
  const hasAttentionMetadata = Boolean(
    model.embeddingLength && model.attentionHeadCount && model.attentionHeadCountKv,
  );
  const kvMibPerLayer = hasAttentionMetadata
    ? Math.ceil(
        (contextSize *
          Math.ceil(model.embeddingLength! / model.attentionHeadCount!) *
          model.attentionHeadCountKv! *
          2 *
          2) /
          MIB,
      )
    : Math.ceil(contextSize / 4096) * 16;

  const devices = loadConfig.gpuLayers.map((allocation) => {
    const node = nodes.find((candidate) => candidate.id === allocation.nodeId);
    const weightMib = Math.ceil((modelMib * allocation.layers) / totalLayers);
    if (allocation.kind === "cpu") {
      const availableVramMib = Math.floor(Math.max(0, node?.ramAvailableGb ?? 0) * 1024);
      return {
        ...allocation,
        kind: "cpu" as const,
        estimatedVramMib: weightMib,
        availableVramMib,
        fits: weightMib <= availableVramMib,
      };
    }
    const availableVramMib = Math.floor(Math.max(0, node?.gpu.vramAvailableGb ?? 0) * 1024);
    const estimatedVramMib = allocation.layers
      ? weightMib + kvMibPerLayer * allocation.layers + GPU_RUNTIME_ALLOWANCE_MIB
      : 0;
    return {
      ...allocation,
      kind: "gpu" as const,
      estimatedVramMib,
      availableVramMib,
      fits: estimatedVramMib <= availableVramMib,
    };
  });
  const cpuLayers = totalLayers - assignedLayers;
  return {
    totalLayers,
    gpuLayers,
    cpuLayers,
    estimatedCpuRamMib: Math.ceil((modelMib * cpuLayers) / totalLayers),
    usesAttentionMetadata: hasAttentionMetadata,
    devices,
  };
}

export function distributeLayersByVram(
  totalLayers: number,
  nodes: NodeCapabilities[],
): ModelLoadConfig["gpuLayers"] {
  if (!nodes.length) return [];
  const capable = nodes.filter((node) => node.online && node.gpu.vramAvailableGb > 0);
  if (!capable.length) return [];
  const totalVram = capable.reduce(
    (total, node) => total + Math.max(0, node.gpu.vramAvailableGb),
    0,
  );
  let assigned = 0;
  return capable.map((node, index) => {
    const layers =
      index === capable.length - 1
        ? totalLayers - assigned
        : Math.floor((totalLayers * Math.max(0, node.gpu.vramAvailableGb)) / totalVram);
    assigned += layers;
    return { nodeId: node.id, layers };
  });
}

export function fitLayersByVram(
  model: ModelRecord,
  nodes: NodeCapabilities[],
): ModelLoadConfig["gpuLayers"] {
  if (!model.layerCount || !nodes.length) return [];
  for (let gpuLayerCount = model.layerCount; gpuLayerCount >= 1; gpuLayerCount -= 1) {
    const gpuLayers = distributeLayersByVram(gpuLayerCount, nodes);
    const estimate = estimateModelSplitLocally(model, nodes, {
      contextSize: 4096,
      gpuLayers,
    });
    if (estimate.devices.every((device) => device.fits)) return gpuLayers;
  }
  return [];
}
