import type { AppService, ChatResponse } from "../types";
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

export const nativeService: AppService = {
  getAppSnapshot: () => backend("get_app_snapshot"),
  completeSetup: (deviceName) => backend("complete_setup", { deviceName }),
  updateSettings: (settings) => backend("update_settings", { settings }),
  installRuntime: async (onProgress) => {
    onProgress?.(100, "Python backend ready");
    return backend("install_runtime");
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
  sendChatMessage: async (messages, settings, images, onStream) => {
    onStream?.({ kind: "status", status: "generating" });
    const response = await backend<ChatResponse>("send_chat_message", { messages, settings, images });
    if (response.content) onStream?.({ kind: "token", content: response.content });
    onStream?.({ kind: "status", status: "idle" });
    return response;
  },
  cancelGeneration: () => backend("cancel_generation"),
  getApiConfig: () => backend("get_api_config"),
  regenerateApiKey: () => backend("regenerate_api_key"),
  openNetworkSettings: () => invoke("open_network_settings"),
  openLogsFolder: () => invoke("open_logs_folder"),
};
