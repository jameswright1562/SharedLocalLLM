import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { render } from "../test/render";
import { cloneSnapshot, serviceWith } from "../test/fixtures";
import type { AppSnapshot, PageProps } from "../types";
import { BenchmarksPage } from "./BenchmarksPage";
import { ModelsPage } from "./ModelsPage";
import { NetworkPage } from "./NetworkPage";
import { NodesPage } from "./NodesPage";
import { OverviewPage } from "./OverviewPage";

function props(snapshot: AppSnapshot = cloneSnapshot(), serviceOverrides = {}): PageProps {
  return {
    snapshot,
    service: serviceWith(snapshot, serviceOverrides),
    refreshSnapshot: vi.fn().mockResolvedValue(undefined),
    navigate: vi.fn(),
  };
}

describe("dashboard pages", () => {
  it("renders active, errored, and empty cluster states", async () => {
    const snapshot = cloneSnapshot();
    snapshot.cluster = {
      status: "running",
      coordinatorNodeId: "node-a",
      workerNodeId: "node-b",
      modelId: "model-text",
      error: "Worker tunnel closed",
    };
    const pageProps = props(snapshot);
    const { rerender } = render(<OverviewPage {...pageProps} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Worker tunnel closed");
    expect(screen.getByText("Orchid 9B Q4_K_M")).toBeInTheDocument();
    const stopCluster = vi.fn().mockResolvedValue({ status: "ready" });
    const runningProps = props(snapshot, { stopCluster });
    rerender(<OverviewPage {...runningProps} />);
    await userEvent.click(screen.getByRole("button", { name: /stop cluster/i }));
    expect(stopCluster).toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: /choose model/i }));
    expect(runningProps.navigate).toHaveBeenCalledWith("models");

    const empty = cloneSnapshot();
    empty.nodes = [];
    empty.network = undefined;
    empty.cluster = { status: "idle" };
    rerender(<OverviewPage {...props(empty)} />);
    expect(screen.getByText(/pair a second computer/i)).toBeInTheDocument();
    expect(screen.getByText(/none loaded/i)).toBeInTheDocument();
    expect(screen.getByText(/untested/i)).toBeInTheDocument();
  });

  it("refreshes node capabilities and shows the missing peer prompt", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.nodes[1]!.online = false;
    snapshot.nodes[1]!.adapter.linkSpeedMbps = undefined;
    const refreshHardware = vi.fn().mockResolvedValue(snapshot);
    const pageProps = props(snapshot, { refreshHardware });
    const { rerender } = render(<NodesPage {...pageProps} />);
    expect(screen.getByText("Offline")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /refresh hardware/i }));
    await waitFor(() => expect(pageProps.refreshSnapshot).toHaveBeenCalled());

    snapshot.nodes = snapshot.nodes.slice(0, 1);
    rerender(<NodesPage {...props(snapshot)} />);
    expect(screen.getByText(/no worker connected/i)).toBeInTheDocument();
  });

  it("connects a worker from the nodes page after a refresh error", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.nodes = snapshot.nodes.slice(0, 1);
    const refreshHardware = vi.fn().mockRejectedValue("probe failed");
    const connectPeer = vi.fn().mockResolvedValue(cloneSnapshot().nodes[1]);
    const pageProps = props(snapshot, { refreshHardware, connectPeer });
    render(<NodesPage {...pageProps} />);

    await user.click(screen.getByRole("button", { name: /refresh hardware/i }));
    expect(await screen.findByRole("status")).toHaveTextContent("probe failed");
    await user.type(screen.getByLabelText(/ethernet ipv4 address/i), "10.10.10.2");
    await user.click(screen.getByRole("button", { name: /^connect$/i }));
    expect(connectPeer).toHaveBeenCalledWith("10.10.10.2");
  });

  it("forgets a paired worker only after explicit confirmation", async () => {
    const user = userEvent.setup();
    const resetPairing = vi.fn().mockResolvedValue({
      ...cloneSnapshot(),
      nodes: cloneSnapshot().nodes.slice(0, 1),
    });
    const pageProps = props(cloneSnapshot(), { resetPairing });
    render(<NodesPage {...pageProps} />);

    await user.click(screen.getByRole("button", { name: /forget remote node/i }));
    expect(resetPairing).not.toHaveBeenCalled();
    expect(screen.getByText(/model files and folders stay untouched/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /confirm forget remote node/i }));
    expect(resetPairing).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(pageProps.refreshSnapshot).toHaveBeenCalled());
  });

  it("covers untested, poor, and failed network diagnostics", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.network = undefined;
    snapshot.nodes = snapshot.nodes.slice(0, 1);
    const { rerender } = render(<NetworkPage {...props(snapshot)} />);
    expect(screen.getByText(/no link result yet/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run network test/i })).toBeDisabled();

    snapshot.nodes = cloneSnapshot().nodes;
    const runNetworkTest = vi.fn().mockRejectedValue("peer disconnected");
    rerender(<NetworkPage {...props(snapshot, { runNetworkTest })} />);
    await user.click(screen.getByRole("button", { name: /run network test/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("peer disconnected");

    const poor = { ...cloneSnapshot().network!, classification: "poor" as const };
    rerender(<NetworkPage key="poor" {...props({ ...snapshot, network: poor })} />);
    expect(screen.getByText(/prefer single-node inference/i)).toBeInTheDocument();
  });

  it("runs model discovery, folder selection, launch, and failure paths", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.models.push({
      ...snapshot.models[0]!,
      id: "too-large",
      name: "Colossus 70B",
      fit: "does-not-fit",
    });
    const discoverModels = vi
      .fn()
      .mockRejectedValueOnce(new Error("Index locked"))
      .mockResolvedValue(snapshot.models);
    const addModelDirectory = vi
      .fn()
      .mockRejectedValueOnce("dialog error")
      .mockResolvedValue(snapshot.modelDirectories[0]);
    const startCluster = vi
      .fn()
      .mockRejectedValueOnce("runtime busy")
      .mockResolvedValue({ status: "loading" });
    const pageProps = props(snapshot, { discoverModels, addModelDirectory, startCluster });
    render(<ModelsPage {...pageProps} />);

    await user.click(screen.getByRole("button", { name: /^refresh$/i }));
    expect(await screen.findByRole("status")).toHaveTextContent("Index locked");
    await user.click(screen.getByRole("button", { name: /^refresh$/i }));
    expect(await screen.findByRole("status")).toHaveTextContent("refreshed");

    await user.click(screen.getByRole("button", { name: /^add folder$/i }));
    expect(await screen.findByRole("status")).toHaveTextContent("dialog error");
    await user.click(screen.getByRole("button", { name: /^add folder$/i }));
    await waitFor(() => expect(pageProps.refreshSnapshot).toHaveBeenCalled());

    await user.click(screen.getByRole("radio", { name: /^split$/i }));
    expect(within(screen.getByTestId("model-list")).getByText(/atlas vision/i)).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /^text$/i }));
    await user.clear(screen.getByRole("searchbox"));
    await user.type(screen.getByRole("searchbox"), "Colossus");
    const colossusRow = within(screen.getByTestId("model-list"))
      .getByText(/colossus 70b/i)
      .closest("tr");
    await user.click(colossusRow as HTMLElement);
    expect(await screen.findByRole("button", { name: /launch colossus/i })).toBeDisabled();

    await user.clear(screen.getByRole("searchbox"));
    await user.type(screen.getByRole("searchbox"), "Orchid");
    const orchidRow = within(screen.getByTestId("model-list"))
      .getByText(/orchid 9b/i)
      .closest("tr");
    await user.click(orchidRow as HTMLElement);
    await user.click(await screen.findByRole("button", { name: /launch orchid/i }));
    expect(await screen.findByRole("status")).toHaveTextContent("runtime busy");
    await user.click(screen.getByRole("button", { name: /launch orchid/i }));
    await waitFor(() => expect(startCluster).toHaveBeenCalledTimes(2));
  });

  it("uses the selected context and treats a cancelled folder dialog as no change", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    const addModelDirectory = vi.fn().mockResolvedValue(null);
    const startCluster = vi.fn().mockResolvedValue({ status: "running", modelId: "model-text" });
    const pageProps = props(snapshot, { addModelDirectory, startCluster });
    render(<ModelsPage {...pageProps} />);

    const orchidRow = within(screen.getByTestId("model-list"))
      .getByText(/orchid 9b/i)
      .closest("tr");
    await user.click(orchidRow as HTMLElement);
    const contextInput = await screen.findByLabelText("Requested context");
    await user.clear(contextInput);
    await user.type(contextInput, "12288");
    await user.click(screen.getByRole("button", { name: /launch orchid/i }));
    expect(startCluster).toHaveBeenCalledWith("model-text", {
      contextSize: 12288,
      gpuLayers: expect.any(Array),
      includeRemoteCpu: false,
      force: false,
      flashAttention: false,
      useMmap: true,
      useMlock: false,
      cpuThreads: 0,
      batchSize: 512,
    });

    await user.click(screen.getByRole("button", { name: /^add folder$/i }));
    expect(await screen.findByRole("status")).toHaveTextContent(/no folder selected/i);
    expect(pageProps.refreshSnapshot).toHaveBeenCalledTimes(1);
  });

  it("relaunches a model with its saved load configuration", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    Object.assign(snapshot.models[0]!, { layerCount: 40 });
    snapshot.modelLoadConfigs = {
      "model-text": {
        contextSize: 4096,
        gpuLayers: [{ nodeId: "node-a", layers: 12 }],
        includeRemoteCpu: false,
        force: false,
        flashAttention: true,
        useMmap: false,
        useMlock: true,
        cpuThreads: 6,
        batchSize: 1024,
      },
    };
    const startCluster = vi.fn().mockResolvedValue({ status: "running", modelId: "model-text" });
    render(<ModelsPage {...props(snapshot, { startCluster })} />);

    const orchidRow = within(screen.getByTestId("model-list"))
      .getByText(/orchid 9b/i)
      .closest("tr");
    await user.click(orchidRow as HTMLElement);

    expect(await screen.findByLabelText("Requested context")).toHaveValue(4096);
    expect(screen.getByRole("checkbox", { name: /flash attention/i })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /lock model in ram/i })).toBeChecked();

    await user.click(screen.getByRole("button", { name: /launch orchid/i }));
    expect(startCluster).toHaveBeenCalledWith("model-text", {
      contextSize: 4096,
      gpuLayers: [{ nodeId: "node-a", layers: 12 }],
      includeRemoteCpu: false,
      force: false,
      flashAttention: true,
      useMmap: false,
      useMlock: true,
      cpuThreads: 6,
      batchSize: 1024,
    });
  });

  it("configures GPU layers per computer and previews estimated VRAM before launch", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    Object.assign(snapshot.models[0]!, { layerCount: 40 });
    const startCluster = vi.fn().mockResolvedValue({ status: "running", modelId: "model-text" });
    render(<ModelsPage {...props(snapshot, { startCluster })} />);

    const orchidRow = within(screen.getByTestId("model-list"))
      .getByText(/orchid 9b/i)
      .closest("tr");
    await user.click(orchidRow as HTMLElement);
    await user.click(await screen.findByRole("radio", { name: /manual gpu split/i }));

    expect(screen.getByRole("heading", { name: /gpu layer allocation/i })).toBeInTheDocument();
    const localLayers = await screen.findByLabelText(/gpu layers on studio host/i);
    const remoteLayers = screen.getByLabelText(/gpu layers on remote node/i);
    await user.clear(localLayers);
    await user.type(localLayers, "24");
    await user.clear(remoteLayers);
    await user.type(remoteLayers, "16");

    expect(screen.getByText(/40 of 40 layers on gpu/i)).toBeInTheDocument();
    expect(screen.getAllByText(/estimated vram/i)).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: /launch orchid/i }));
    expect(startCluster).toHaveBeenCalledWith("model-text", {
      contextSize: 8192,
      gpuLayers: [
        { nodeId: "node-a", layers: 24 },
        { nodeId: "node-b", layers: 16 },
      ],
      includeRemoteCpu: false,
      force: false,
      flashAttention: false,
      useMmap: true,
      useMlock: false,
      cpuThreads: 0,
      batchSize: 512,
    });
  });

  it("edits saved GPU layer shares that carry an explicit gpu kind tag", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    Object.assign(snapshot.models[0]!, { layerCount: 40 });
    snapshot.modelLoadConfigs = {
      "model-text": {
        contextSize: 8192,
        gpuLayers: [
          { nodeId: "node-a", layers: 30, kind: "gpu" },
          { nodeId: "node-b", layers: 10, kind: "gpu" },
        ],
        includeRemoteCpu: false,
        force: false,
        flashAttention: false,
        useMmap: true,
        useMlock: false,
        cpuThreads: 0,
        batchSize: 512,
      },
    };
    const startCluster = vi.fn().mockResolvedValue({ status: "running", modelId: "model-text" });
    render(<ModelsPage {...props(snapshot, { startCluster })} />);

    const orchidRow = within(screen.getByTestId("model-list"))
      .getByText(/orchid 9b/i)
      .closest("tr");
    await user.click(orchidRow as HTMLElement);

    const localLayers = await screen.findByLabelText(/gpu layers on studio host/i);
    await user.clear(localLayers);
    await user.type(localLayers, "24");
    expect(screen.getByText(/34 of 40 layers on gpu/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /launch orchid/i }));
    expect(startCluster).toHaveBeenCalledWith(
      "model-text",
      expect.objectContaining({
        gpuLayers: [
          { nodeId: "node-a", layers: 24, kind: "gpu" },
          { nodeId: "node-b", layers: 10, kind: "gpu" },
        ],
      }),
    );
  });

  it("offloads a layer share to the worker's CPU when requested", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    Object.assign(snapshot.models[0]!, { layerCount: 40 });
    const startCluster = vi.fn().mockResolvedValue({ status: "running", modelId: "model-text" });
    render(<ModelsPage {...props(snapshot, { startCluster })} />);

    const orchidRow = within(screen.getByTestId("model-list"))
      .getByText(/orchid 9b/i)
      .closest("tr");
    await user.click(orchidRow as HTMLElement);
    await user.click(await screen.findByRole("radio", { name: /manual gpu split/i }));

    await user.click(screen.getByRole("checkbox", { name: /offload layers to remote node/i }));
    const cpuLayers = await screen.findByLabelText(/cpu layers on remote node/i);
    expect(cpuLayers).toBeInTheDocument();
    await user.clear(cpuLayers);
    await user.type(cpuLayers, "4");

    await user.click(screen.getByRole("button", { name: /launch orchid/i }));
    expect(startCluster).toHaveBeenCalledWith(
      "model-text",
      expect.objectContaining({
        includeRemoteCpu: true,
        gpuLayers: expect.arrayContaining([{ nodeId: "node-b", layers: 4, kind: "cpu" }]),
      }),
    );
  });

  it("describes advanced load options and forwards their values on launch", async () => {
    const user = userEvent.setup();
    const startCluster = vi.fn().mockResolvedValue({ status: "running", modelId: "model-text" });
    render(<ModelsPage {...props(cloneSnapshot(), { startCluster })} />);

    const orchidRow = within(screen.getByTestId("model-list"))
      .getByText(/orchid 9b/i)
      .closest("tr");
    await user.click(orchidRow as HTMLElement);

    expect(
      await screen.findByRole("heading", { name: /advanced load options/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/uses flash attention to accelerate generation/i)).toBeInTheDocument();
    expect(screen.getByText(/maps the model file into memory/i)).toBeInTheDocument();
    expect(screen.getByText(/prevents windows from swapping/i)).toBeInTheDocument();
    expect(screen.getByText(/0 lets llama.cpp choose/i)).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: /flash attention/i }));
    await user.click(screen.getByRole("checkbox", { name: /lock model in ram/i }));
    const threads = screen.getByRole("spinbutton", { name: /cpu threads/i });
    await user.type(threads, "{Control>}a{/Control}8");
    const batch = screen.getByRole("spinbutton", { name: /batch size/i });
    await user.type(batch, "{Control>}a{/Control}2048");

    expect(
      screen.getByText(/unsupported gpus fall back to standard attention/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/needs free ram equal to the whole model/i)).toBeInTheDocument();
    expect(screen.getByText(/can oversubscribe the cpu/i)).toBeInTheDocument();
    expect(screen.getByText(/very large batches use more memory/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /launch orchid/i }));
    expect(startCluster).toHaveBeenCalledWith("model-text", {
      contextSize: 8192,
      gpuLayers: expect.any(Array),
      includeRemoteCpu: false,
      force: false,
      flashAttention: true,
      useMmap: true,
      useMlock: true,
      cpuThreads: 8,
      batchSize: 2048,
    });
  });

  it("allows force launch for a model that does not fit and passes the flag through", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.models.push({
      ...snapshot.models[0]!,
      id: "too-large",
      name: "Colossus 70B",
      fit: "does-not-fit",
    });
    const startCluster = vi.fn().mockResolvedValue({ status: "loading", modelId: "too-large" });
    render(<ModelsPage {...props(snapshot, { startCluster })} />);

    await user.type(screen.getByRole("searchbox"), "Colossus");
    const colossusRow = within(screen.getByTestId("model-list"))
      .getByText(/colossus 70b/i)
      .closest("tr");
    await user.click(colossusRow as HTMLElement);
    const launchButton = await screen.findByRole("button", { name: /launch colossus/i });
    expect(launchButton).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: /force launch/i }));
    expect(launchButton).toBeEnabled();

    await user.click(launchButton);
    expect(startCluster).toHaveBeenCalledWith("too-large", {
      contextSize: 8192,
      gpuLayers: expect.any(Array),
      includeRemoteCpu: false,
      force: true,
      flashAttention: false,
      useMmap: true,
      useMlock: false,
      cpuThreads: 0,
      batchSize: 512,
    });
  });

  it("renders benchmark results and routes an empty benchmark action", async () => {
    const pageProps = props();
    const { rerender } = render(<BenchmarksPage {...pageProps} />);
    expect(screen.getByText("Recommended")).toBeInTheDocument();
    expect(screen.getByText(/120.5/)).toBeInTheDocument();

    const empty = cloneSnapshot();
    empty.benchmarks = [];
    const emptyProps = props(empty);
    rerender(<BenchmarksPage key="empty" {...emptyProps} />);
    expect(screen.getByText(/no benchmark runs/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /benchmark a model/i }));
    expect(emptyProps.navigate).toHaveBeenCalledWith("models");
  });

  it("benchmarks a running model in place instead of planning a reload", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.cluster = { status: "running", coordinatorNodeId: "node-a", modelId: "model-text" };
    const runInferenceBenchmark = vi.fn().mockResolvedValue([]);
    const pageProps = props(snapshot, { runInferenceBenchmark });
    const view = render(<BenchmarksPage {...pageProps} />);

    expect(screen.getByText(/benchmarks the loaded instance/i)).toBeInTheDocument();
    expect(screen.queryByText(/automatic gpu split/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /run benchmark/i }));
    expect(runInferenceBenchmark).toHaveBeenCalledWith("model-text");

    const peerRunning = cloneSnapshot();
    peerRunning.cluster = { status: "ready" };
    peerRunning.nodes[1]!.clusterStatus = "running";
    peerRunning.nodes[1]!.clusterModelId = "model-text";
    view.rerender(<BenchmarksPage key="peer" {...props(peerRunning, { runInferenceBenchmark })} />);
    expect(screen.getByText(/benchmarks the loaded instance/i)).toBeInTheDocument();
  });

  it("renders benchmark run times from epoch seconds and tolerates invalid values", () => {
    const snapshot = cloneSnapshot();
    const base = cloneSnapshot().benchmarks[0]!;
    snapshot.benchmarks = [
      { ...base, id: "epoch-run", ranAt: String(Math.floor(Date.now() / 1000)) },
      { ...base, id: "broken-run", ranAt: "" },
    ];
    render(<BenchmarksPage {...props(snapshot)} />);
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.getAllByText("—")).toHaveLength(1);
  });

  it("benchmarks a running model in place instead of planning a reload", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.cluster = { status: "running", coordinatorNodeId: "node-a", modelId: "model-text" };
    const runInferenceBenchmark = vi.fn().mockResolvedValue([]);
    const pageProps = props(snapshot, { runInferenceBenchmark });
    const view = render(<BenchmarksPage {...pageProps} />);

    expect(screen.getByText(/benchmarks the loaded instance/i)).toBeInTheDocument();
    expect(screen.queryByText(/automatic gpu split/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /run benchmark/i }));
    expect(runInferenceBenchmark).toHaveBeenCalledWith("model-text");

    const peerRunning = cloneSnapshot();
    peerRunning.cluster = { status: "ready" };
    peerRunning.nodes[1]!.clusterStatus = "running";
    peerRunning.nodes[1]!.clusterModelId = "model-text";
    view.rerender(<BenchmarksPage key="peer" {...props(peerRunning, { runInferenceBenchmark })} />);
    expect(screen.getByText(/benchmarks the loaded instance/i)).toBeInTheDocument();
  });

  it("runs and cancels inference benchmarks through the native service", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.benchmarks = [];
    snapshot.models[1]!.layerCount = 40;
    let resolveBenchmark!: (rows: AppSnapshot["benchmarks"]) => void;
    const runInferenceBenchmark = vi
      .fn()
      .mockImplementation(() => new Promise((resolve) => (resolveBenchmark = resolve)));
    const cancelInferenceBenchmark = vi.fn().mockResolvedValue(undefined);
    render(
      <BenchmarksPage {...props(snapshot, { runInferenceBenchmark, cancelInferenceBenchmark })} />,
    );

    await user.selectOptions(screen.getByLabelText(/benchmark model/i), "model-vision");
    expect(screen.getByRole("status")).toHaveTextContent(/studio host: 25 layers/i);
    expect(screen.getByRole("status")).toHaveTextContent(/remote node: 15 layers/i);
    await user.click(screen.getByRole("button", { name: /run benchmark/i }));
    expect(runInferenceBenchmark).toHaveBeenCalledWith("model-vision");
    await user.click(screen.getByRole("button", { name: /cancel benchmark/i }));
    expect(cancelInferenceBenchmark).toHaveBeenCalled();
    resolveBenchmark(cloneSnapshot().benchmarks);
    await Promise.resolve();
    expect(screen.queryByText("Recommended")).not.toBeInTheDocument();
  });

  it("shows actionable inference benchmark errors", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.benchmarks = [];
    const runInferenceBenchmark = vi.fn().mockRejectedValue({
      code: "benchmark_runtime_missing",
      message: "llama-bench.exe is not installed.",
      action: "Install or repair the pinned runtime.",
    });
    render(<BenchmarksPage {...props(snapshot, { runInferenceBenchmark })} />);

    await user.click(screen.getByRole("button", { name: /run benchmark/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/install or repair/i);
  });
});
