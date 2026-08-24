import type {
  AppliedModelTune,
  AppService,
  AppSnapshot,
  AutotuneStatus,
  ChatMessage,
  ChatResponse,
  ChatSettings,
  ChatStreamEvent,
} from "../types";
import { decodeAppError } from "./errors";

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
  try {
    return await tauriInvoke<T>(command, args);
  } catch (reason) {
    throw decodeAppError(reason);
  }
}

async function backend<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
  return invoke<T>("backend_request", { command, args });
}

type StreamPayload =
  | { type: "reasoning"; content: string }
  | { type: "token"; content: string }
  | { type: "stats"; tokensPerSecond: number }
  | { type: "done" }
  | { type: "error"; message: string };

async function streamChatCompletion(
  messages: ChatMessage[],
  settings: ChatSettings,
  images: string[],
  onStream?: (event: ChatStreamEvent) => void,
): Promise<ChatResponse> {
  const { invoke: tauriInvoke, Channel } = await import("@tauri-apps/api/core");
  const channel = new Channel<StreamPayload>();
  channel.onmessage = (payload) => {
    switch (payload.type) {
      case "reasoning":
        onStream?.({ kind: "reasoning", content: payload.content });
        break;
      case "token":
        onStream?.({ kind: "token", content: payload.content });
        break;
      case "stats":
        onStream?.({ kind: "stats", tokensPerSecond: payload.tokensPerSecond });
        break;
      case "done":
        onStream?.({ kind: "status", status: "idle" });
        break;
    }
  };
  try {
    return await tauriInvoke<ChatResponse>("backend_stream", {
      command: "send_chat_message",
      args: { messages, settings, images },
      channel,
    });
  } catch (reason) {
    throw decodeAppError(reason);
  }
}

export const nativeService: AppService = {
  getAppSnapshot: () => backend("get_app_snapshot"),
  completeSetup: (deviceName) => backend("complete_setup", { deviceName }),
  updateSettings: (settings) => backend("update_settings", { settings }),
  installRuntime: async (onProgress) => {
    onProgress?.(25, "Verifying the Python and llama.cpp backend");
    const snapshot = await backend<AppSnapshot>("install_runtime");
    onProgress?.(100, "Python backend verified");
    return snapshot;
  },
  refreshHardware: () => backend("refresh_hardware"),
  discoverModels: () => backend("discover_models"),
  addModelDirectory: async () => {
    const path = await invoke<string | null>("pick_model_directory");
    return path ? backend("add_model_directory", { path }) : null;
  },
  removeModelDirectory: (id) => backend("remove_model_directory", { id }),
  runNetworkTest: () => backend("run_network_test"),
  connectPeer: (manualEndpoint) => backend("connect_peer", { manualEndpoint }),
  resetPairing: () => backend("reset_pairing"),
  estimateModelSplit: (modelId, loadConfig) =>
    backend("estimate_model_split", { modelId, loadConfig }),
  startCluster: (modelId, loadConfig) => backend("start_cluster", { modelId, loadConfig }),
  stopCluster: () => backend("stop_cluster"),
  runInferenceBenchmark: (modelId) => backend("run_inference_benchmark", { modelId }),
  cancelInferenceBenchmark: () => backend("cancel_inference_benchmark"),
  startModelAutotune: (modelId, depth) =>
    backend<AutotuneStatus>("start_model_autotune", { modelId, depth }),
  getAutotuneStatus: () => backend<AutotuneStatus>("get_autotune_status"),
  cancelModelAutotune: () => backend("cancel_model_autotune"),
  applyModelTune: (modelId) => backend<AppliedModelTune>("apply_model_tune", { modelId }),
  sendChatMessage: async (messages, settings, images, onStream) => {
    onStream?.({ kind: "status", status: "processing" });
    let receivedContent = false;
    try {
      return await streamChatCompletion(messages, settings, images, (event) => {
        if (event.kind === "token" || event.kind === "reasoning") receivedContent = true;
        onStream?.(event);
      });
    } catch (reason) {
      if (receivedContent) throw reason;
      onStream?.({ kind: "status", status: "generating" });
      const response = await backend<ChatResponse>("send_chat_message", {
        messages,
        settings,
        images,
      });
      if (response.reasoning) onStream?.({ kind: "reasoning", content: response.reasoning });
      if (response.content) onStream?.({ kind: "token", content: response.content });
      if (response.tokensPerSecond !== undefined) {
        onStream?.({ kind: "stats", tokensPerSecond: response.tokensPerSecond });
      }
      onStream?.({ kind: "status", status: "idle" });
      return response;
    }
  },
  cancelGeneration: () => backend("cancel_generation"),
  getApiConfig: () => backend("get_api_config"),
  regenerateApiKey: () => backend("regenerate_api_key"),
  tryApiRequest: () => backend("try_api_request"),
  openNetworkSettings: () => invoke("open_network_settings"),
  openLogsFolder: () => invoke("open_logs_folder"),
};
