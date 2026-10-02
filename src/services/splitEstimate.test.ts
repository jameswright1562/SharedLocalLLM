import { describe, expect, it } from "vitest";
import type { ModelRecord, NodeCapabilities } from "../types";
import {
  distributeLayersByVram,
  estimateModelSplitLocally,
  normalizeManualGpuLayers,
  upsertGpuNodeLayers,
} from "./splitEstimate";

function node(id: string, availableVram: number, ram = 32): NodeCapabilities {
  return {
    id,
    name: id,
    online: true,
    role: id === "local" ? "coordinator" : "worker",
    cpu: "CPU",
    ramTotalGb: ram,
    ramAvailableGb: ram,
    gpu: { name: "GPU", vramTotalGb: 8, vramAvailableGb: availableVram },
    adapter: { name: "Ethernet", kind: "ethernet" },
  };
}

const model: ModelRecord = {
  id: "model",
  name: "Model",
  architecture: "llama",
  quantization: "Q4_K_M",
  sizeBytes: 4 * 1024 ** 3,
  contextLength: 8192,
  layerCount: 32,
  capability: "text",
  shards: 1,
  locations: [],
  fit: "single-node",
  isLocal: true,
};

describe("split estimates", () => {
  it("does not allocate layers to a GPU with no available VRAM", () => {
    expect(distributeLayersByVram(32, [node("empty", 0), node("local", 8)])).toEqual([
      { nodeId: "local", layers: 32 },
    ]);
  });

  it("counts explicit remote CPU layers toward the model total", () => {
    const estimate = estimateModelSplitLocally(model, [node("local", 8), node("peer", 8)], {
      contextSize: 4096,
      includeRemoteCpu: true,
      gpuLayers: [
        { nodeId: "local", layers: 20 },
        { nodeId: "peer", layers: 12, kind: "cpu" },
      ],
    });
    expect(estimate.gpuLayers).toBe(20);
    expect(estimate.cpuLayers).toBe(0);
  });

  it("rejects a combined GPU and remote CPU over-allocation", () => {
    expect(() =>
      estimateModelSplitLocally(model, [node("local", 8), node("peer", 8)], {
        contextSize: 4096,
        includeRemoteCpu: true,
        gpuLayers: [
          { nodeId: "local", layers: 24 },
          { nodeId: "peer", layers: 16, kind: "cpu" },
        ],
      }),
    ).toThrow("selects 40 layers");
  });

  it("uses a finite context fallback for incomplete remote metadata", () => {
    const incomplete = { ...model, contextLength: undefined as unknown as number };
    const estimate = estimateModelSplitLocally(incomplete, [node("local", 8)], {
      contextSize: 4096,
      gpuLayers: [{ nodeId: "local", layers: 16 }],
    });
    expect(Number.isFinite(estimate.devices[0]?.estimatedVramMib)).toBe(true);
  });

  it("adds a late-joining GPU node with zero layers instead of leaving it uneditable", () => {
    const stale = [{ nodeId: "local", layers: 24 }];
    const normalized = normalizeManualGpuLayers(stale, [node("local", 8), node("peer", 8)]);
    expect(normalized).toEqual([
      { nodeId: "local", layers: 24 },
      { nodeId: "peer", layers: 0 },
    ]);
    const estimate = estimateModelSplitLocally(model, [node("local", 8), node("peer", 8)], {
      contextSize: 4096,
      gpuLayers: normalized,
    });
    expect(estimate.devices.map((device) => device.nodeId).sort()).toEqual(["local", "peer"]);
  });

  it("preserves remote CPU and offline entries while ensuring every GPU node exists", () => {
    const normalized = normalizeManualGpuLayers(
      [
        { nodeId: "local", layers: 10 },
        { nodeId: "peer", layers: 4, kind: "cpu" },
        { nodeId: "gone", layers: 2 },
      ],
      [node("local", 8), node("peer", 8)],
    );
    expect(normalized).toEqual([
      { nodeId: "local", layers: 10 },
      { nodeId: "peer", layers: 0 },
      { nodeId: "peer", layers: 4, kind: "cpu" },
      { nodeId: "gone", layers: 2 },
    ]);
  });

  it("upserts GPU layers for a node with no existing allocation entry", () => {
    expect(upsertGpuNodeLayers([{ nodeId: "local", layers: 24 }], "peer", "10", 65)).toEqual([
      { nodeId: "local", layers: 24 },
      { nodeId: "peer", layers: 10 },
    ]);
    expect(upsertGpuNodeLayers([{ nodeId: "local", layers: 24 }], "local", "12", 65)).toEqual([
      { nodeId: "local", layers: 12 },
    ]);
    expect(upsertGpuNodeLayers([], "peer", "999", 65)).toEqual([{ nodeId: "peer", layers: 65 }]);
    expect(upsertGpuNodeLayers([], "peer", "not-a-number", 65)).toEqual([
      { nodeId: "peer", layers: 0 },
    ]);
  });
});
