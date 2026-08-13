import type { AppService, ChatMessage, ChatResponse, ChatSettings } from "../types";
import { decodeAppError } from "./errors";

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
  try {
    return await tauriInvoke<T>(command, args);
  } catch (reason) {
    throw decodeAppError(reason);
  }
}

export const nativeService: AppService = {
  getAppSnapshot: () => invoke("get_app_snapshot"),
  completeSetup: (deviceName) => invoke("complete_setup", { deviceName }),
  updateSettings: (settings) => invoke("update_settings", { settings }),
  installRuntime: async (onProgress) => {
    let unlisten: (() => void) | undefined;
    if (onProgress) {
      const { listen } = await import("@tauri-apps/api/event");
      unlisten = await listen<{ percent: number; status: string }>("runtime-progress", (event) =>
        onProgress(event.payload.percent, event.payload.status),
      );
    }
    try {
      return await invoke("install_runtime");
    } finally {
      unlisten?.();
    }
  },
  refreshHardware: () => invoke("refresh_hardware"),
  discoverModels: () => invoke("discover_models"),
  addModelDirectory: () => invoke("add_model_directory"),
  removeModelDirectory: (id) => invoke("remove_model_directory", { id }),
  runNetworkTest: () => invoke("run_network_test"),
  generatePairingCode: (allowPublicNetwork = false) =>
    invoke("generate_pairing_code", { allowPublicNetwork }),
  pairWithPeer: (code, allowPublicNetwork = false, manualEndpoint) =>
    invoke("pair_with_peer", { code, allowPublicNetwork, manualEndpoint }),
  resetPairing: () => invoke("reset_pairing"),
  estimateModelSplit: (modelId, loadConfig) =>
    invoke("estimate_model_split", { modelId, loadConfig }),
  startCluster: (modelId, loadConfig) => invoke("start_cluster", { modelId, loadConfig }),
  stopCluster: () => invoke("stop_cluster"),
  runInferenceBenchmark: (modelId) => invoke("run_inference_benchmark", { modelId }),
  cancelInferenceBenchmark: () => invoke("cancel_inference_benchmark"),
  sendChatMessage: (messages: ChatMessage[], settings: ChatSettings, images: string[]) =>
    invoke<ChatResponse>("send_chat_message", { messages, settings, images }),
  cancelGeneration: () => invoke("cancel_generation"),
  getApiConfig: () => invoke("get_api_config"),
  regenerateApiKey: () => invoke("regenerate_api_key"),
  openNetworkSettings: () => invoke("open_network_settings"),
  openLogsFolder: () => invoke("open_logs_folder"),
};
