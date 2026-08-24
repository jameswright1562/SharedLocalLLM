import type {
  ApiTryResult,
  AppService,
  AutotuneStatus,
  ModelDirectory,
  ModelLoadConfig,
  ModelTuneResult,
  NetworkBenchmark,
} from "../types";
import { demoApi, demoNodes, demoSnapshot } from "./demoData";
import { estimateModelSplitLocally } from "./splitEstimate";

const delay = (ms = 180) => new Promise((resolve) => window.setTimeout(resolve, ms));

let demoTune: AutotuneStatus = { status: "idle" };

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
    demoSnapshot.authRequired = settings.authRequired;
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
  async connectPeer() {
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
  async startCluster(modelId, loadConfig) {
    await delay(900);
    demoSnapshot.modelLoadConfigs = {
      ...demoSnapshot.modelLoadConfigs,
      [modelId]: structuredClone(loadConfig),
    };
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
  async startModelAutotune(modelId, depth) {
    await delay(150);
    const model = demoSnapshot.models.find((item) => item.id === modelId);
    if (!model) throw new Error("The tuning model is unavailable.");
    if (demoSnapshot.cluster.status === "running") {
      throw new Error("Stop the running cluster before tuning this model.");
    }
    const stages = [
      "batch",
      "gpu-layers",
      "threads",
      ...(depth === "full" ? ["tensor-split", "kv-cache", "op-offload", "poll"] : []),
      "final-verification",
    ];
    demoTune = {
      status: "running",
      modelId,
      modelName: model.name,
      depth,
      stageIndex: 0,
      stageCount: stages.length,
      currentStage: null,
      events: [],
      result: null,
    };
    stages.forEach((stage, index) => {
      window.setTimeout(
        () => {
          if (demoTune.status !== "running") return;
          demoTune = {
            ...demoTune,
            stageIndex: index + 1,
            currentStage: stage,
            events: [
              ...(demoTune.events ?? []),
              { type: "stage-result", stage, bestTokensPerSecond: 92 + index * 2.5 },
            ],
          };
          if (index === stages.length - 1) {
            const result: ModelTuneResult = {
              modelId,
              modelName: model.name,
              depth,
              ranAt: new Date().toISOString(),
              fingerprint: "demo-topology",
              winners: { batchSize: 2048, uBatch: 512, cpuThreads: 8, kvCacheK: "q4_0" },
              promptTokensPerSecond: 118.6,
              generationTokensPerSecond: 24.1,
            };
            demoTune = { ...demoTune, status: "complete", currentStage: null, result };
            demoSnapshot.modelTunes = { ...demoSnapshot.modelTunes, [modelId]: result };
          }
        },
        450 * (index + 1),
      );
    });
    return structuredClone(demoTune);
  },
  async getAutotuneStatus() {
    await delay(60);
    return structuredClone(demoTune);
  },
  async cancelModelAutotune() {
    await delay(60);
    if (demoTune.status === "running") demoTune = { ...demoTune, status: "cancelled" };
  },
  async applyModelTune(modelId) {
    await delay(160);
    const tune = demoSnapshot.modelTunes?.[modelId];
    if (!tune) throw new Error("This model has no saved tuning result. Run Auto-tune first.");
    const base: ModelLoadConfig = demoSnapshot.modelLoadConfigs?.[modelId] ?? {
      contextSize: 4096,
      gpuLayers: [],
    };
    const loadConfig = {
      ...base,
      batchSize: tune.winners.batchSize ?? base.batchSize ?? 512,
      uBatch: tune.winners.uBatch ?? base.uBatch,
      cpuThreads: tune.winners.cpuThreads ?? base.cpuThreads,
    };
    demoSnapshot.modelLoadConfigs = { ...demoSnapshot.modelLoadConfigs, [modelId]: loadConfig };
    return { loadConfig: structuredClone(loadConfig), staleTopology: false, tunedAt: tune.ranAt };
  },
  async sendChatMessage(messages, _settings, _images, onStream) {
    await delay(600);
    const latest = [...messages].reverse().find((message) => message.role === "user");
    const topic = latest?.content ? ` about “${latest.content.slice(0, 52)}”` : "";
    const reasoning = `I traced the request${topic} across both nodes and re-checked the model placement before answering.`;
    const content = `The cluster received your request${topic}. This browser preview simulates the streamed response; native mode routes it through the active coordinator.`;
    if (onStream) {
      onStream({ kind: "status", status: "processing" });
      for (const word of reasoning.split(" ")) {
        await delay(12);
        onStream({ kind: "reasoning", content: `${word} ` });
      }
      for (const word of content.split(" ")) {
        await delay(14);
        onStream({ kind: "token", content: `${word} ` });
      }
      onStream({ kind: "stats", tokensPerSecond: 24.8 });
      onStream({ kind: "status", status: "idle" });
    }
    return { content, reasoning, tokensPerSecond: 24.8 };
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
  async tryApiRequest(): Promise<ApiTryResult> {
    await delay(900);
    if (demoSnapshot.cluster.status !== "running") {
      return {
        status: 400,
        durationMs: 4,
        body: JSON.stringify({
          error: {
            message: "No model is loaded. Start a cluster, then try the request again.",
            type: "model_not_loaded",
          },
        }),
      };
    }
    const model = demoSnapshot.models.find((item) => item.id === demoSnapshot.cluster.modelId);
    return {
      status: 200,
      durationMs: 640,
      body: JSON.stringify({
        id: "chatcmpl-sharedlocalllm",
        object: "chat.completion",
        model: model?.id ?? "active",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content:
                "This browser preview simulates the response. Native mode returns the completion produced by the loaded model.",
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 9, completion_tokens: 24, total_tokens: 33 },
      }),
    };
  },
  async openNetworkSettings() {
    await delay(80);
  },
  async openLogsFolder() {
    await delay(80);
  },
};
