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
}

export interface GpuLayerAllocation {
  nodeId: string;
  layers: number;
}

export interface ModelLoadConfig {
  contextSize: number;
  gpuLayers: GpuLayerAllocation[];
  force?: boolean;
}

export interface DeviceVramEstimate {
  nodeId: string;
  layers: number;
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
  error?: string;
}

export interface AppSnapshot {
  setupComplete: boolean;
  runtime: { status: RuntimeStatus; version?: string; error?: string };
  deviceName: string;
  nodes: NodeCapabilities[];
  models: ModelRecord[];
  modelDirectories: ModelDirectory[];
  network?: NetworkBenchmark;
  cluster: ClusterSession;
  benchmarks: InferenceBenchmark[];
  logs: string[];
  apiPort: number;
  autostart: boolean;
}

export interface AppSettings {
  deviceName: string;
  apiPort: number;
  autostart: boolean;
}

export interface ApiConfig {
  url: string;
  apiKey: string;
  healthy: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
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
}

export type ChatStreamEvent =
  { kind: "token"; content: string } | { kind: "status"; status: string };

export interface AppService {
  getAppSnapshot(): Promise<AppSnapshot>;
  completeSetup(deviceName: string): Promise<AppSnapshot>;
  updateSettings(settings: AppSettings): Promise<AppSnapshot>;
  installRuntime(onProgress?: (percent: number, status: string) => void): Promise<AppSnapshot>;
  refreshHardware(): Promise<AppSnapshot>;
  discoverModels(): Promise<ModelRecord[]>;
  addModelDirectory(): Promise<ModelDirectory | null>;
  removeModelDirectory(id: string): Promise<void>;
  runNetworkTest(): Promise<NetworkBenchmark>;
  connectPeer(manualEndpoint?: string): Promise<NodeCapabilities>;
  resetPairing(): Promise<AppSnapshot>;
  estimateModelSplit(modelId: string, loadConfig: ModelLoadConfig): Promise<SplitEstimate>;
  startCluster(modelId: string, loadConfig: ModelLoadConfig): Promise<ClusterSession>;
  stopCluster(): Promise<ClusterSession>;
  runInferenceBenchmark(modelId: string): Promise<InferenceBenchmark[]>;
  cancelInferenceBenchmark(): Promise<void>;
  sendChatMessage(
    messages: ChatMessage[],
    settings: ChatSettings,
    images: string[],
    onStream?: (event: ChatStreamEvent) => void,
  ): Promise<ChatResponse>;
  cancelGeneration(): Promise<void>;
  getApiConfig(): Promise<ApiConfig>;
  regenerateApiKey(): Promise<ApiConfig>;
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
