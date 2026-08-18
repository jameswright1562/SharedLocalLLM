import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

import { appService, demoService, nativeService } from "./appService";

describe("app services", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    invokeMock
      .mockReset()
      .mockImplementation((command: string, payload?: Record<string, unknown>) => {
        if (command === "pick_model_directory") return Promise.resolve("C:\\Models");
        if (command === "backend_request") {
          const request = payload as { command?: string } | undefined;
          if (request?.command === "send_chat_message")
            return Promise.resolve({ content: "hello" });
        }
        return Promise.resolve({ status: "ready" });
      });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function settle<T>(promise: Promise<T>): Promise<T> {
    await vi.runAllTimersAsync();
    return promise;
  }

  it("provides a complete deterministic browser service", async () => {
    expect(appService).toBe(demoService);
    const initial = await settle(demoService.getAppSnapshot());
    expect(initial.nodes).toHaveLength(2);
    expect(await settle(demoService.refreshHardware())).toMatchObject({
      deviceName: "Primary node",
    });
    expect(await settle(demoService.discoverModels())).toHaveLength(3);

    const completed = await settle(demoService.completeSetup("Demo coordinator"));
    expect(completed).toMatchObject({ setupComplete: true, deviceName: "Demo coordinator" });
    const updated = await settle(
      demoService.updateSettings({
        deviceName: "Renamed coordinator",
        apiPort: 12000,
        autostart: true,
      }),
    );
    expect(updated).toMatchObject({
      deviceName: "Renamed coordinator",
      apiPort: 12000,
      autostart: true,
    });

    const progress = vi.fn();
    const installed = await settle(demoService.installRuntime(progress));
    expect(progress).toHaveBeenLastCalledWith(100, "Runtime ready");
    expect(installed.runtime.status).toBe("ready");

    const directory = await settle(demoService.addModelDirectory());
    expect(directory?.path).toContain("Custom");
    await settle(demoService.removeModelDirectory(directory!.id));
    expect((await settle(demoService.getAppSnapshot())).modelDirectories).not.toContainEqual(
      directory,
    );

    expect((await settle(demoService.runNetworkTest())).classification).toBe("good");
    expect((await settle(demoService.connectPeer())).role).toBe("worker");

    const demoLoadConfig = { contextSize: 4096, gpuLayers: [] };
    expect(await settle(demoService.startCluster("meridian-12b", demoLoadConfig))).toMatchObject({
      status: "running",
      modelId: "meridian-12b",
    });
    expect((await settle(demoService.stopCluster())).status).toBe("ready");
    expect(await settle(demoService.runInferenceBenchmark("meridian-12b"))).toEqual([
      expect.objectContaining({ modelName: "Meridian 12B Instruct", recommended: true }),
    ]);
  });

  it("routes backend operations through one Tauri bridge", async () => {
    const progress = vi.fn();
    await nativeService.getAppSnapshot();
    await nativeService.completeSetup("Main node");
    await nativeService.updateSettings({
      deviceName: "Main node",
      apiPort: 11435,
      autostart: true,
    });
    await nativeService.installRuntime(progress);
    expect(progress).toHaveBeenCalledWith(100, "Python backend ready");
    await nativeService.refreshHardware();
    await nativeService.discoverModels();
    await nativeService.addModelDirectory();
    await nativeService.removeModelDirectory("dir-7");
    await nativeService.runNetworkTest();
    await nativeService.connectPeer("192.168.50.2");
    await nativeService.resetPairing();
    const loadConfig = {
      contextSize: 8192,
      gpuLayers: [
        { nodeId: "node-a", layers: 24 },
        { nodeId: "node-b", layers: 16 },
      ],
    };
    await nativeService.estimateModelSplit("model-2", loadConfig);
    await nativeService.startCluster("model-2", loadConfig);
    await nativeService.stopCluster();
    await nativeService.runInferenceBenchmark("model-2");
    await nativeService.cancelInferenceBenchmark();
    const stream = vi.fn();
    await nativeService.sendChatMessage(
      [{ id: "m1", role: "user", content: "hello" }],
      { systemPrompt: "", temperature: 0.4, maxTokens: 256 },
      [],
      stream,
    );
    expect(stream).toHaveBeenCalledWith({ kind: "token", content: "hello" });
    await nativeService.cancelGeneration();
    await nativeService.getApiConfig();
    await nativeService.regenerateApiKey();
    await nativeService.openNetworkSettings();
    await nativeService.openLogsFolder();

    const backendCommands = invokeMock.mock.calls
      .filter(([command]) => command === "backend_request")
      .map(([, value]) => (value as { command: string }).command);
    expect(backendCommands).toEqual([
      "get_app_snapshot",
      "complete_setup",
      "update_settings",
      "install_runtime",
      "refresh_hardware",
      "discover_models",
      "add_model_directory",
      "remove_model_directory",
      "run_network_test",
      "connect_peer",
      "reset_pairing",
      "estimate_model_split",
      "start_cluster",
      "stop_cluster",
      "run_inference_benchmark",
      "cancel_inference_benchmark",
      "send_chat_message",
      "cancel_generation",
      "get_api_config",
      "regenerate_api_key",
    ]);
    expect(invokeMock).toHaveBeenCalledWith("pick_model_directory", undefined);
    expect(invokeMock).toHaveBeenCalledWith("open_network_settings", undefined);
    expect(invokeMock).toHaveBeenCalledWith("open_logs_folder", undefined);
  });

  it("decodes structured backend errors through the shared adapter", async () => {
    invokeMock.mockRejectedValueOnce({
      code: "api_port_in_use",
      message: "127.0.0.1:11435 is already in use.",
      action: "Choose another local API port.",
    });
    await expect(nativeService.getAppSnapshot()).rejects.toMatchObject({
      code: "api_port_in_use",
      message: "127.0.0.1:11435 is already in use.",
      action: "Choose another local API port.",
    });
  });
});
