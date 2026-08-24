import type { ReactNode } from "react";

export type PageId =
  "overview" | "nodes" | "network" | "models" | "benchmarks" | "chat" | "api" | "settings";

export type RuntimeStatus = "ready" | "missing" | "installing" | "error";
export type ClusterStatus =
  "idle" | "pairing" | "loading" | "ready" | "running" | "stopping" | "error";
export type NodeRole = "coordinator" | "worker" | "available";
export type ModelFit = "single-node" | "combined-gpu" | "gpu-ram" | "does-not-fit";

export interface GpuCapabilities {
  name: string;
  vramTotalGb: number;
  vramAvailableGb: number;
  driverVersion?: string;
}

export interface NodeCapabilities {
  id: string;
  name: string;
  online: boolean;
  role: NodeRole;
  cpu: string;
  ramTotalGb: number;
  ramAvailableGb: number;
  gpu: GpuCapabilities;
  adapter: {
    name: string;
    kind: "ethernet" | "wifi" | "other";
    linkSpeedMbps?: number;
  };
  clusterStatus?: string;
  clusterModelId?: string;
}

export interface ModelLocation {
  nodeId: string;
  path: string;
  source: "lm-studio" | "custom";
}

export interface ModelRecord {
  id: string;
  name: string;
  architecture: string;
  quantization: string;
  sizeBytes: number;
  contextLength: number;
  layerCount?: number;
  embeddingLength?: number;
  attentionHeadCount?: number;
  attentionHeadCountKv?: number;
  capability: "text" | "vision";
  shards: number;
  locations: ModelLocation[];
  fit: ModelFit;
  remoteOnly?: boolean;
  isLocal: boolean;
  mtp?: boolean;
  reasoningPreserve?: boolean;
}

export interface GpuLayerAllocation {
  nodeId: string;
  layers: number;
  kind?: "gpu" | "cpu";
}

export interface ModelLoadConfig {
  contextSize: number;
  gpuLayers: GpuLayerAllocation[];
  includeRemoteCpu?: boolean;
  force?: boolean;
  flashAttention?: boolean;
  useMmap?: boolean;
  useMlock?: boolean;
  cpuThreads?: number;
  batchSize?: number;
  engine?: "builtin" | "llama-server";
  uBatch?: number;
  kvCacheK?: string;
  kvCacheV?: string;
  noOpOffload?: number;
  rpcPoll?: number;
}

export interface ModelTuneWinners {
  batchSize?: number | null;
  uBatch?: number | null;
  cpuThreads?: number | null;
  kvCacheK?: string;
  kvCacheV?: string;
  gpuLayers?: number | null;
  tensorSplit?: number[] | null;
  noOpOffload?: number | null;
  poll?: number | null;
  gpuLayersAllocations?: GpuLayerAllocation[] | null;
}

export interface ModelTuneResult {
  modelId: string;
  modelName: string;
  depth: "quick" | "full";
  ranAt: string;
  fingerprint: string;
  winners: ModelTuneWinners;
  promptTokensPerSecond: number;
  generationTokensPerSecond: number;
}

export interface AutotuneEvent {
  type: string;
  stage?: string;
  label?: string;
  bestTokensPerSecond?: number;
}

export type AutotuneRunStatus = "idle" | "running" | "complete" | "failed" | "cancelled";

export interface AutotuneStatus {
  status: AutotuneRunStatus;
  modelId?: string;
  modelName?: string;
  depth?: "quick" | "full";
  stageIndex?: number;
  stageCount?: number;
  currentStage?: string | null;
  error?: string;
  result?: ModelTuneResult | null;
  events?: AutotuneEvent[];
}

export interface AppliedModelTune {
  loadConfig: ModelLoadConfig;
  staleTopology: boolean;
  tunedAt?: string;
}

export interface ModelLoadOptions {
  flashAttention: boolean;
  useMmap: boolean;
  useMlock: boolean;
  cpuThreads: number;
  batchSize: number;
}

export interface DeviceVramEstimate {
  nodeId: string;
  layers: number;
  kind?: "gpu" | "cpu";
  estimatedVramMib: number;
  availableVramMib: number;
  fits: boolean;
}

export interface SplitEstimate {
  totalLayers: number;
  gpuLayers: number;
  cpuLayers: number;
  estimatedCpuRamMib: number;
  usesAttentionMetadata: boolean;
  devices: DeviceVramEstimate[];
}

export interface ModelDirectory {
  id: string;
  nodeId: string;
  path: string;
  source: "lm-studio" | "custom";
}

export interface NetworkBenchmark {
  downMbps: number;
  upMbps: number;
  latencyMedianMs: number;
  latencyP95Ms: number;
  jitterMs: number;
  packetLossPercent: number;
  classification: "good" | "usable" | "poor";
  adapter: string;
  windowsProfile?: string;
}

export interface InferenceBenchmark {
  id: string;
  modelName: string;
  topology: "local" | "remote" | "distributed";
  gpuLayers?: GpuLayerAllocation[];
  promptTokensPerSecond: number;
  generationTokensPerSecond: number;
  loadTimeSeconds: number;
  memoryPeakGb: number;
  recommended: boolean;
  ranAt: string;
  error?: string;
}

export interface ClusterSession {
  status: ClusterStatus;
  coordinatorNodeId?: string;
  workerNodeId?: string;
  modelId?: string;
  engine?: "builtin" | "llama-server";
  error?: string;
}

export interface AppSnapshot {
  setupComplete: boolean;
  runtime: { status: RuntimeStatus; version?: string; error?: string };
  deviceName: string;
  deviceId?: string;
  nodes: NodeCapabilities[];
  models: ModelRecord[];
  modelDirectories: ModelDirectory[];
  network?: NetworkBenchmark;
  cluster: ClusterSession;
  modelLoadConfigs?: Record<string, ModelLoadConfig>;
  modelTunes?: Record<string, ModelTuneResult>;
  benchmarks: InferenceBenchmark[];
  logs: string[];
  apiPort: number;
  authRequired: boolean;
  autostart: boolean;
}

export interface AppSettings {
  deviceName: string;
  apiPort: number;
  authRequired: boolean;
  autostart: boolean;
}

export interface ApiConfig {
  url: string;
  apiKey: string;
  authRequired: boolean;
  healthy: boolean;
}

export interface ApiTryResult {
  status: number;
  durationMs: number;
  body: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  reasoning?: string;
  tokensPerSecond?: number;
  imageNames?: string[];
  imageData?: string[];
  error?: boolean;
}

export interface ChatSettings {
  systemPrompt: string;
  temperature: number;
  maxTokens: number;
}

export interface ChatResponse {
  content: string;
  reasoning?: string;
  tokensPerSecond?: number;
}

export type ChatStreamEvent =
  | { kind: "status"; status: string }
  | { kind: "token"; content: string }
  | { kind: "reasoning"; content: string }
  | { kind: "stats"; tokensPerSecond: number }
  | { kind: "done" };

export interface AppService {
  getAppSnapshot(): Promise<AppSnapshot>;
  completeSetup(deviceName: string): Promise<AppSnapshot>;
  updateSettings(settings: AppSettings): Promise<AppSnapshot>;
  installRuntime(onProgress?: (percent: number, status: string) => void): Promise<AppSnapshot>;
  refreshHardware(): Promise<AppSnapshot>;
  discoverModels(): Promise<ModelRecord[]>;
  addModelDirectory(): Promise<ModelDirectory | null>;
  removeModelDirectory(id: string): Promise<void>;
  deleteModelFolder(folder: string): Promise<void>;
  runNetworkTest(): Promise<NetworkBenchmark>;
  connectPeer(manualEndpoint?: string): Promise<NodeCapabilities>;
  resetPairing(): Promise<AppSnapshot>;
  estimateModelSplit(modelId: string, loadConfig: ModelLoadConfig): Promise<SplitEstimate>;
  startCluster(modelId: string, loadConfig: ModelLoadConfig): Promise<ClusterSession>;
  stopCluster(): Promise<ClusterSession>;
  runInferenceBenchmark(modelId: string): Promise<InferenceBenchmark[]>;
  cancelInferenceBenchmark(): Promise<void>;
  startModelAutotune(modelId: string, depth: "quick" | "full"): Promise<AutotuneStatus>;
  getAutotuneStatus(): Promise<AutotuneStatus>;
  cancelModelAutotune(): Promise<void>;
  applyModelTune(modelId: string): Promise<AppliedModelTune>;
  sendChatMessage(
    messages: ChatMessage[],
    settings: ChatSettings,
    images: string[],
    onStream?: (event: ChatStreamEvent) => void,
  ): Promise<ChatResponse>;
  cancelGeneration(): Promise<void>;
  getApiConfig(): Promise<ApiConfig>;
  regenerateApiKey(): Promise<ApiConfig>;
  tryApiRequest(): Promise<ApiTryResult>;
  openNetworkSettings(): Promise<void>;
  openLogsFolder(): Promise<void>;
}

export interface PageProps {
  snapshot: AppSnapshot;
  service: AppService;
  refreshSnapshot: () => Promise<void>;
  navigate: (page: PageId) => void;
}

export interface IconProps {
  name: string;
  children?: ReactNode;
}
