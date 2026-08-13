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

  const gpuLayers = loadConfig.gpuLayers.reduce((total, item) => total + item.layers, 0);
  if (gpuLayers > totalLayers)
    throw new Error(`The split selects ${gpuLayers} GPU layers, but the model has ${totalLayers}.`);

  const contextSize = Math.max(4096, Math.min(loadConfig.contextSize, model.contextLength));
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
    const availableVramMib = Math.floor(Math.max(0, node?.gpu.vramAvailableGb ?? 0) * 1024);
    const weightMib = Math.ceil((modelMib * allocation.layers) / totalLayers);
    const estimatedVramMib = allocation.layers
      ? weightMib + kvMibPerLayer * allocation.layers + GPU_RUNTIME_ALLOWANCE_MIB
      : 0;
    return {
      ...allocation,
      estimatedVramMib,
      availableVramMib,
      fits: estimatedVramMib <= availableVramMib,
    };
  });
  const cpuLayers = totalLayers - gpuLayers;
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
  const reservedPerNode = totalLayers >= nodes.length ? 1 : 0;
  const distributable = Math.max(0, totalLayers - reservedPerNode * nodes.length);
  const totalVram = nodes.reduce((total, node) => total + Math.max(0, node.gpu.vramAvailableGb), 0);
  let assigned = 0;
  return nodes.map((node, index) => {
    const layers =
      index === nodes.length - 1
        ? totalLayers - assigned
        : totalVram > 0
          ? reservedPerNode +
            Math.floor((distributable * Math.max(0, node.gpu.vramAvailableGb)) / totalVram)
          : reservedPerNode + Math.floor(distributable / nodes.length);
    assigned += layers;
    return { nodeId: node.id, layers };
  });
}

export function fitLayersByVram(
  model: ModelRecord,
  nodes: NodeCapabilities[],
): ModelLoadConfig["gpuLayers"] {
  if (!model.layerCount || !nodes.length) return [];
  for (let gpuLayerCount = model.layerCount; gpuLayerCount >= nodes.length; gpuLayerCount -= 1) {
    const gpuLayers = distributeLayersByVram(gpuLayerCount, nodes);
    const estimate = estimateModelSplitLocally(model, nodes, {
      contextSize: 4096,
      gpuLayers,
    });
    if (estimate.devices.every((device) => device.fits)) return gpuLayers;
  }
  return [];
}
