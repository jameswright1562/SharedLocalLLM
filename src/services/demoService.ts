import type { AppService, ModelDirectory, NetworkBenchmark } from "../types";
import { demoApi, demoNodes, demoSnapshot } from "./demoData";
import { estimateModelSplitLocally } from "./splitEstimate";

const delay = (ms = 180) => new Promise((resolve) => window.setTimeout(resolve, ms));

export const demoService: AppService = {
  async getAppSnapshot() {
    await delay();
    return structuredClone(demoSnapshot);
  },
  async completeSetup(deviceName) {
    await delay();
    demoSnapshot.deviceName = deviceName.trim();
    demoSnapshot.nodes[0]!.name = demoSnapshot.deviceName;
    demoSnapshot.setupComplete = true;
    return structuredClone(demoSnapshot);
  },
  async updateSettings(settings) {
    await delay();
    demoSnapshot.deviceName = settings.deviceName.trim();
    demoSnapshot.nodes[0]!.name = demoSnapshot.deviceName;
    demoSnapshot.apiPort = settings.apiPort;
    demoSnapshot.autostart = settings.autostart;
    return structuredClone(demoSnapshot);
  },
  async installRuntime(onProgress) {
    demoSnapshot.runtime = { status: "installing" };
    for (const [percent, status] of [
      [15, "Downloading runtime"],
      [54, "Verifying archive"],
      [82, "Installing CUDA libraries"],
      [100, "Runtime ready"],
    ] as Array<[number, string]>) {
      await delay(220);
      onProgress?.(percent, status);
    }
    demoSnapshot.runtime = { status: "ready", version: "llama.cpp b6123" };
    return structuredClone(demoSnapshot);
  },
  async refreshHardware() {
    await delay(420);
    return structuredClone(demoSnapshot);
  },
  async discoverModels() {
    await delay(460);
    return structuredClone(demoSnapshot.models);
  },
  async addModelDirectory() {
    await delay();
    const directory: ModelDirectory = {
      id: `directory-${Date.now()}`,
      nodeId: "local-node",
      path: "D:\\Models\\Custom",
      source: "custom",
    };
    demoSnapshot.modelDirectories.push(directory);
    return directory;
  },
  async removeModelDirectory(id) {
    await delay();
    demoSnapshot.modelDirectories = demoSnapshot.modelDirectories.filter(
      (directory) => directory.id !== id,
    );
  },
  async runNetworkTest() {
    await delay(850);
    return structuredClone(demoSnapshot.network as NetworkBenchmark);
  },
  async generatePairingCode() {
    await delay();
    return { code: "482 916", expiresInSeconds: 300 };
  },
  async pairWithPeer() {
    await delay(620);
    const peerNode = demoNodes[1]!;
    if (!demoSnapshot.nodes.some((node) => node.id === "peer-node"))
      demoSnapshot.nodes.push(peerNode);
    return structuredClone(peerNode);
  },
  async resetPairing() {
    await delay();
    demoSnapshot.nodes = demoSnapshot.nodes.slice(0, 1);
    demoSnapshot.network = undefined;
    demoSnapshot.cluster = { status: "idle" };
    return structuredClone(demoSnapshot);
  },
  async estimateModelSplit(modelId, loadConfig) {
    await delay(80);
    const model = demoSnapshot.models.find((item) => item.id === modelId);
    if (!model) throw new Error("The model is unavailable.");
    return estimateModelSplitLocally(model, demoSnapshot.nodes, loadConfig);
  },
  async startCluster(modelId) {
    await delay(900);
    demoSnapshot.cluster = {
      status: "running",
      coordinatorNodeId: "local-node",
      workerNodeId: "peer-node",
      modelId,
    };
    return structuredClone(demoSnapshot.cluster);
  },
  async stopCluster() {
    await delay(320);
    demoSnapshot.cluster = { ...demoSnapshot.cluster, status: "ready", modelId: undefined };
    return structuredClone(demoSnapshot.cluster);
  },
  async runInferenceBenchmark(modelId) {
    await delay(780);
    const model = demoSnapshot.models.find((item) => item.id === modelId);
    if (!model) throw new Error("The benchmark model is unavailable.");
    const result = {
      id: `benchmark-${Date.now()}`,
      modelName: model.name,
      topology: model.fit === "combined-gpu" ? ("distributed" as const) : ("local" as const),
      promptTokensPerSecond: 126.4,
      generationTokensPerSecond: 21.6,
      loadTimeSeconds: 7.8,
      memoryPeakGb: model.sizeBytes / 1_000_000_000,
      recommended: true,
      ranAt: new Date().toISOString(),
    };
    demoSnapshot.benchmarks = [result, ...demoSnapshot.benchmarks];
    return [structuredClone(result)];
  },
  async cancelInferenceBenchmark() {
    await delay(80);
  },
  async sendChatMessage(messages) {
    await delay(950);
    const latest = [...messages].reverse().find((message) => message.role === "user");
    return {
      content: `The cluster received your request${latest?.content ? ` about “${latest.content.slice(0, 52)}”` : ""}. This browser preview simulates the streamed response; native mode routes it through the active coordinator.`,
    };
  },
  async cancelGeneration() {
    await delay(80);
  },
  async getApiConfig() {
    await delay(80);
    return structuredClone(demoApi);
  },
  async regenerateApiKey() {
    await delay(200);
    demoApi.apiKey = `sk-local-${crypto.randomUUID().replaceAll("-", "").slice(0, 16)}`;
    return structuredClone(demoApi);
  },
  async openNetworkSettings() {
    await delay(80);
  },
  async openLogsFolder() {
    await delay(80);
  },
};
