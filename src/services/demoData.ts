import type { ApiConfig, AppSnapshot, ModelRecord, NodeCapabilities } from "../types";

export const demoNodes: NodeCapabilities[] = [
  {
    id: "local-node",
    name: "Primary node",
    online: true,
    role: "coordinator",
    cpu: "12-core processor",
    ramTotalGb: 48,
    ramAvailableGb: 34.6,
    gpu: {
      name: "CUDA device 0",
      vramTotalGb: 16,
      vramAvailableGb: 13.8,
      driverVersion: "Current",
    },
    adapter: { name: "2.5 GbE", kind: "ethernet", linkSpeedMbps: 2500 },
  },
  {
    id: "peer-node",
    name: "Compute node",
    online: true,
    role: "worker",
    cpu: "8-core processor",
    ramTotalGb: 32,
    ramAvailableGb: 25.2,
    gpu: {
      name: "CUDA device 1",
      vramTotalGb: 12,
      vramAvailableGb: 10.4,
      driverVersion: "Current",
    },
    adapter: { name: "1 GbE", kind: "ethernet", linkSpeedMbps: 1000 },
  },
];

export const demoModels: ModelRecord[] = [
  {
    id: "meridian-12b",
    name: "Meridian 12B Instruct",
    architecture: "llama",
    quantization: "Q5_K_M",
    sizeBytes: 9_120_000_000,
    contextLength: 32768,
    layerCount: 40,
    embeddingLength: 5120,
    attentionHeadCount: 40,
    attentionHeadCountKv: 8,
    capability: "text",
    shards: 1,
    locations: [
      { nodeId: "local-node", path: "C:\\Models\\meridian-12b-q5.gguf", source: "lm-studio" },
    ],
    fit: "single-node",
  },
  {
    id: "northstar-27b",
    name: "Northstar 27B",
    architecture: "qwen2",
    quantization: "Q4_K_M",
    sizeBytes: 18_800_000_000,
    contextLength: 32768,
    layerCount: 64,
    embeddingLength: 5120,
    attentionHeadCount: 40,
    attentionHeadCountKv: 8,
    capability: "text",
    shards: 2,
    locations: [
      { nodeId: "peer-node", path: "D:\\AI\\northstar-00001-of-00002.gguf", source: "custom" },
    ],
    fit: "combined-gpu",
  },
  {
    id: "spectra-vision",
    name: "Spectra Vision 11B",
    architecture: "qwen2vl",
    quantization: "Q4_K_M",
    sizeBytes: 8_700_000_000,
    contextLength: 16384,
    layerCount: 40,
    embeddingLength: 3584,
    attentionHeadCount: 28,
    attentionHeadCountKv: 4,
    capability: "vision",
    shards: 1,
    locations: [
      { nodeId: "local-node", path: "C:\\Models\\spectra-vision.gguf", source: "custom" },
    ],
    fit: "single-node",
  },
];

const baseSnapshot: AppSnapshot = {
  setupComplete: true,
  runtime: { status: "ready", version: "llama.cpp b6123" },
  deviceName: "Primary node",
  nodes: demoNodes,
  models: demoModels,
  modelDirectories: [
    {
      id: "lm-studio",
      nodeId: "local-node",
      path: "%USERPROFILE%\\.lmstudio\\models",
      source: "lm-studio",
    },
    { id: "custom-models", nodeId: "peer-node", path: "D:\\AI", source: "custom" },
  ],
  network: {
    downMbps: 934,
    upMbps: 917,
    latencyMedianMs: 1.2,
    latencyP95Ms: 2.6,
    jitterMs: 0.4,
    packetLossPercent: 0,
    classification: "good",
    adapter: "Ethernet · full duplex",
  },
  cluster: {
    status: "ready",
    coordinatorNodeId: "local-node",
    workerNodeId: "peer-node",
  },
  benchmarks: [
    {
      id: "benchmark-1",
      modelName: "Northstar 27B",
      topology: "distributed",
      promptTokensPerSecond: 112.4,
      generationTokensPerSecond: 18.7,
      loadTimeSeconds: 10.4,
      memoryPeakGb: 20.1,
      recommended: true,
      ranAt: new Date().toISOString(),
    },
    {
      id: "benchmark-2",
      modelName: "Meridian 12B Instruct",
      topology: "local",
      promptTokensPerSecond: 164.2,
      generationTokensPerSecond: 27.8,
      loadTimeSeconds: 6.8,
      memoryPeakGb: 11.2,
      recommended: false,
      ranAt: new Date(Date.now() - 86_400_000).toISOString(),
    },
  ],
  logs: [
    "14:22:08  INFO  Peer channel authenticated",
    "14:22:09  INFO  RPC tunnel bound to loopback",
    "14:22:10  INFO  Model catalogue indexed: 3 models",
    "14:22:11  READY Local API listening on 127.0.0.1:11435",
  ],
  apiPort: 11435,
  autostart: false,
};

export const demoSnapshot = structuredClone(baseSnapshot);
export const demoApi: ApiConfig = {
  url: "http://127.0.0.1:11435",
  apiKey: "sk-local-demo-8f3d7a19",
  healthy: true,
};
