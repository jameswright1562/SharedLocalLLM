import { vi } from "vitest";

import { demoService } from "../services/appService";
import type { AppService, AppSnapshot } from "../types";

export const snapshotFixture: AppSnapshot = {
  setupComplete: true,
  runtime: { status: "ready", version: "b6123" },
  deviceName: "Studio host",
  nodes: [
    {
      id: "node-a",
      name: "Studio host",
      online: true,
      role: "coordinator",
      cpu: "12-core processor",
      ramTotalGb: 48,
      ramAvailableGb: 36,
      gpu: {
        name: "CUDA GPU A",
        vramTotalGb: 16,
        vramAvailableGb: 14,
      },
      adapter: {
        name: "Ethernet",
        kind: "ethernet",
        linkSpeedMbps: 2500,
      },
    },
    {
      id: "node-b",
      name: "Remote node",
      online: true,
      role: "worker",
      cpu: "8-core processor",
      ramTotalGb: 32,
      ramAvailableGb: 25,
      gpu: {
        name: "CUDA GPU B",
        vramTotalGb: 10,
        vramAvailableGb: 8,
      },
      adapter: { name: "Wi-Fi", kind: "wifi", linkSpeedMbps: 866 },
    },
  ],
  models: [
    {
      id: "model-text",
      name: "Orchid 9B Q4_K_M",
      architecture: "llama",
      quantization: "Q4_K_M",
      sizeBytes: 6_400_000_000,
      contextLength: 32768,
      capability: "text",
      shards: 1,
      locations: [
        {
          nodeId: "node-a",
          path: "D:\\Models\\orchid.gguf",
          source: "custom",
        },
      ],
      fit: "single-node",
    },
    {
      id: "model-vision",
      name: "Atlas Vision 12B",
      architecture: "qwen",
      quantization: "Q5_K_M",
      sizeBytes: 10_700_000_000,
      contextLength: 16384,
      capability: "vision",
      shards: 2,
      locations: [
        {
          nodeId: "node-b",
          path: "E:\\Models\\atlas-00001.gguf",
          source: "lm-studio",
        },
      ],
      fit: "combined-gpu",
    },
  ],
  modelDirectories: [
    {
      id: "dir-1",
      nodeId: "node-a",
      path: "D:\\Models",
      source: "custom",
    },
  ],
  network: {
    downMbps: 932,
    upMbps: 901,
    latencyMedianMs: 1.1,
    latencyP95Ms: 2.2,
    jitterMs: 0.3,
    packetLossPercent: 0,
    classification: "good",
    adapter: "Ethernet",
  },
  cluster: {
    status: "ready",
    coordinatorNodeId: "node-a",
    workerNodeId: "node-b",
  },
  benchmarks: [
    {
      id: "run-1",
      modelName: "Orchid 9B Q4_K_M",
      topology: "local",
      promptTokensPerSecond: 120.5,
      generationTokensPerSecond: 22.2,
      loadTimeSeconds: 4.6,
      memoryPeakGb: 8.4,
      recommended: true,
      ranAt: "2026-08-13T10:00:00.000Z",
    },
  ],
  logs: ["Peer channel ready", "Runtime verified"],
  apiPort: 11435,
  authRequired: true,
  autostart: false,
};

export function cloneSnapshot(): AppSnapshot {
  return structuredClone(snapshotFixture);
}

export function serviceWith(
  snapshot: AppSnapshot = cloneSnapshot(),
  overrides: Partial<AppService> = {},
): AppService {
  return {
    ...demoService,
    getAppSnapshot: vi.fn().mockResolvedValue(snapshot),
    getApiConfig: vi.fn().mockResolvedValue({
      url: "http://127.0.0.1:11435",
      apiKey: "sk-local-1234567890",
      authRequired: true,
      healthy: true,
    }),
    ...overrides,
  };
}
