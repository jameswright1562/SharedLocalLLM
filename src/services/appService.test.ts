import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();
const unlistenMock = vi.fn();
const listenMock = vi
  .fn()
  .mockImplementation(
    (
      _event: string,
      callback: (event: { payload: { percent: number; status: string } }) => void,
    ) => {
      callback({ payload: { percent: 42, status: "Downloading" } });
      return Promise.resolve(unlistenMock);
    },
  );

vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({ listen: listenMock }));

import { appService, demoService, nativeService } from "./appService";

describe("app services", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    invokeMock.mockReset().mockResolvedValue({ status: "ready" });
    listenMock.mockClear();
    unlistenMock.mockClear();
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
    expect((await settle(demoService.generatePairingCode(true))).code).toMatch(/\d{3} \d{3}/);
    expect((await settle(demoService.pairWithPeer("123456", true))).role).toBe("worker");

    expect(
      await settle(
        demoService.estimateModelSplit("meridian-12b", {
          contextSize: 4096,
          gpuLayers: [
            { nodeId: "local-node", layers: 24 },
            { nodeId: "peer-node", layers: 16 },
          ],
        }),
      ),
    ).toMatchObject({ totalLayers: 40, gpuLayers: 40, cpuLayers: 0 });

    const demoLoadConfig = { contextSize: 4096, gpuLayers: [] };
    expect(await settle(demoService.startCluster("meridian-12b", demoLoadConfig))).toMatchObject({
      status: "running",
      modelId: "meridian-12b",
    });
    expect((await settle(demoService.stopCluster())).status).toBe("ready");
    expect(await settle(demoService.runInferenceBenchmark("meridian-12b"))).toEqual([
      expect.objectContaining({ modelName: "Meridian 12B Instruct", recommended: true }),
    ]);
    await settle(demoService.cancelInferenceBenchmark());
    const missingBenchmark = demoService.runInferenceBenchmark("missing-model");
    const missingExpectation = expect(missingBenchmark).rejects.toThrow("unavailable");
    await vi.runAllTimersAsync();
    await missingExpectation;
    expect(
      (
        await settle(
          demoService.sendChatMessage(
            [{ id: "m1", role: "user", content: "memory planning" }],
            { systemPrompt: "Helpful", temperature: 0.5, maxTokens: 100 },
            [],
          ),
        )
      ).content,
    ).toContain("memory planning");
    expect(
      (
        await settle(
          demoService.sendChatMessage(
            [{ id: "m2", role: "system", content: "No user message" }],
            { systemPrompt: "Helpful", temperature: 0.5, maxTokens: 100 },
            [],
          ),
        )
      ).content,
    ).toContain("received your request");
    await settle(demoService.cancelGeneration());

    const api = await settle(demoService.getApiConfig());
    const regenerated = await settle(demoService.regenerateApiKey());
    expect(regenerated.apiKey).not.toBe(api.apiKey);
    await settle(demoService.openNetworkSettings());
    await settle(demoService.openLogsFolder());
  });

  it("maps every native service operation to its exact Tauri command", async () => {
    const progress = vi.fn();
    await nativeService.getAppSnapshot();
    await nativeService.completeSetup("Main node");
    await nativeService.updateSettings({
      deviceName: "Main node",
      apiPort: 11435,
      autostart: true,
    });
    await nativeService.installRuntime(progress);
    expect(progress).toHaveBeenCalledWith(42, "Downloading");
    expect(unlistenMock).toHaveBeenCalled();
    await nativeService.refreshHardware();
    await nativeService.discoverModels();
    await nativeService.addModelDirectory();
    await nativeService.removeModelDirectory("dir-7");
    await nativeService.runNetworkTest();
    await nativeService.generatePairingCode(true);
    await nativeService.pairWithPeer("481209", true, "192.168.50.2");
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
    const messages = [{ id: "m1", role: "user" as const, content: "hello" }];
    const settings = { systemPrompt: "", temperature: 0.4, maxTokens: 256 };
    await nativeService.sendChatMessage(messages, settings, ["image-data"]);
    await nativeService.cancelGeneration();
    await nativeService.getApiConfig();
    await nativeService.regenerateApiKey();
    await nativeService.openNetworkSettings();
    await nativeService.openLogsFolder();

    expect(invokeMock.mock.calls).toEqual([
      ["get_app_snapshot", undefined],
      ["complete_setup", { deviceName: "Main node" }],
      [
        "update_settings",
        {
          settings: { deviceName: "Main node", apiPort: 11435, autostart: true },
        },
      ],
      ["install_runtime", undefined],
      ["refresh_hardware", undefined],
      ["discover_models", undefined],
      ["add_model_directory", undefined],
      ["remove_model_directory", { id: "dir-7" }],
      ["run_network_test", undefined],
      ["generate_pairing_code", { allowPublicNetwork: true }],
      [
        "pair_with_peer",
        { code: "481209", allowPublicNetwork: true, manualEndpoint: "192.168.50.2" },
      ],
      ["reset_pairing", undefined],
      ["estimate_model_split", { modelId: "model-2", loadConfig }],
      ["start_cluster", { modelId: "model-2", loadConfig }],
      ["stop_cluster", undefined],
      ["run_inference_benchmark", { modelId: "model-2" }],
      ["cancel_inference_benchmark", undefined],
      ["send_chat_message", { messages, settings, images: ["image-data"] }],
      ["cancel_generation", undefined],
      ["get_api_config", undefined],
      ["regenerate_api_key", undefined],
      ["open_network_settings", undefined],
      ["open_logs_folder", undefined],
    ]);
  });

  it("removes the runtime listener even when installation fails", async () => {
    invokeMock.mockRejectedValueOnce(new Error("download failed"));
    await expect(nativeService.installRuntime(vi.fn())).rejects.toThrow("download failed");
    expect(unlistenMock).toHaveBeenCalled();
  });

  it("decodes structured Rust command errors through the shared adapter", async () => {
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
