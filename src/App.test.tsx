import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { render } from "./test/render";
import App from "./App";
import { demoService } from "./services/appService";
import type { AppService, AppSnapshot } from "./types";

const readySnapshot: AppSnapshot = {
  setupComplete: true,
  runtime: { status: "ready", version: "b6123" },
  deviceName: "Studio host",
  nodes: [
    {
      id: "node-a",
      name: "Studio host",
      online: true,
      role: "coordinator",
      cpu: "12-core processor",
      ramTotalGb: 48,
      ramAvailableGb: 36,
      gpu: { name: "CUDA GPU A", vramTotalGb: 16, vramAvailableGb: 14 },
      adapter: { name: "Ethernet", kind: "ethernet", linkSpeedMbps: 2500 },
    },
    {
      id: "node-b",
      name: "Remote node",
      online: true,
      role: "worker",
      cpu: "8-core processor",
      ramTotalGb: 32,
      ramAvailableGb: 25,
      gpu: { name: "CUDA GPU B", vramTotalGb: 10, vramAvailableGb: 8 },
      adapter: { name: "Wi-Fi", kind: "wifi", linkSpeedMbps: 866 },
    },
  ],
  models: [
    {
      id: "model-text",
      name: "Orchid 9B Q4_K_M",
      architecture: "llama",
      quantization: "Q4_K_M",
      sizeBytes: 6_400_000_000,
      contextLength: 32768,
      capability: "text",
      shards: 1,
      locations: [{ nodeId: "node-a", path: "D:\\Models\\orchid.gguf", source: "custom" }],
      fit: "single-node",
    },
    {
      id: "model-vision",
      name: "Atlas Vision 12B",
      architecture: "qwen",
      quantization: "Q5_K_M",
      sizeBytes: 10_700_000_000,
      contextLength: 16384,
      capability: "vision",
      shards: 2,
      locations: [{ nodeId: "node-b", path: "E:\\Models\\atlas-00001.gguf", source: "lm-studio" }],
      fit: "combined-gpu",
    },
  ],
  modelDirectories: [{ id: "dir-1", nodeId: "node-a", path: "D:\\Models", source: "custom" }],
  network: {
    downMbps: 932,
    upMbps: 901,
    latencyMedianMs: 1.1,
    latencyP95Ms: 2.2,
    jitterMs: 0.3,
    packetLossPercent: 0,
    classification: "good",
    adapter: "Ethernet",
  },
  cluster: {
    status: "ready",
    coordinatorNodeId: "node-a",
    workerNodeId: "node-b",
  },
  benchmarks: [],
  logs: ["Peer channel ready", "Runtime verified"],
  apiPort: 11435,
  authRequired: true,
  autostart: false,
};

function serviceWith(snapshot: AppSnapshot, overrides: Partial<AppService> = {}): AppService {
  return {
    ...demoService,
    getAppSnapshot: vi.fn().mockResolvedValue(snapshot),
    getApiConfig: vi.fn().mockResolvedValue({
      url: "http://127.0.0.1:11435",
      apiKey: "sk-local-1234567890",
      authRequired: true,
      healthy: true,
    }),
    ...overrides,
  };
}

describe("SharedLocalLLM app", () => {
  it("navigates between the instrument panels", async () => {
    const user = userEvent.setup();
    render(<App service={serviceWith(readySnapshot)} />);

    expect(await screen.findByRole("heading", { name: /cluster overview/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /overview/i })).toHaveAttribute(
      "aria-current",
      "page",
    );
    await user.click(screen.getByRole("button", { name: /network/i }));
    expect(screen.getByRole("heading", { name: /link diagnostics/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /api/i }));
    expect(screen.getByRole("heading", { name: /local api/i })).toBeInTheDocument();
  });

  it("shows useful empty states for unpaired nodes and models", async () => {
    const user = userEvent.setup();
    const empty = { ...readySnapshot, nodes: readySnapshot.nodes.slice(0, 1), models: [] };
    render(<App service={serviceWith(empty)} />);

    expect(await screen.findByText(/pair a second computer/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /models/i }));
    expect(screen.getByText(/no models indexed/i)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /add folder/i }).length).toBeGreaterThan(0);
  });

  it("filters and selects a model while preserving fit labels", async () => {
    const user = userEvent.setup();
    render(<App service={serviceWith(readySnapshot)} />);
    await screen.findByText(/cluster overview/i);
    await user.click(screen.getByRole("button", { name: /models/i }));

    await user.type(screen.getByRole("searchbox", { name: /search models/i }), "vision");
    expect(screen.queryByText(/orchid/i)).not.toBeInTheDocument();
    const row = within(screen.getByTestId("model-list"))
      .getByText(/atlas vision/i)
      .closest("tr");
    expect(row).not.toBeNull();
    await user.click(row as HTMLElement);
    expect(await screen.findByRole("button", { name: /launch atlas vision/i })).toBeEnabled();
    expect(screen.getByText(/combined gpu/i)).toBeInTheDocument();
  });

  it("renders network classification and reruns the benchmark", async () => {
    const user = userEvent.setup();
    const runNetworkTest = vi.fn().mockResolvedValue({
      ...readySnapshot.network,
      classification: "usable",
    });
    render(<App service={serviceWith(readySnapshot, { runNetworkTest })} />);
    await screen.findByText(/cluster overview/i);
    await user.click(screen.getByRole("button", { name: /network/i }));

    expect(screen.getByText(/good/i, { selector: ".classification" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /run network test/i }));
    await waitFor(() => expect(runNetworkTest).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/usable/i, { selector: ".classification" })).toBeInTheDocument();
  });

  it("connects to the other computer during setup", async () => {
    const user = userEvent.setup();
    const setupSnapshot = {
      ...readySnapshot,
      setupComplete: false,
      nodes: readySnapshot.nodes.slice(0, 1),
    };
    const connectPeer = vi.fn().mockResolvedValue(readySnapshot.nodes[1]);
    render(<App service={serviceWith(setupSnapshot, { connectPeer })} />);

    expect(await screen.findByRole("heading", { name: /name this computer/i })).toBeInTheDocument();
    await user.clear(screen.getByLabelText(/device name/i));
    await user.type(screen.getByLabelText(/device name/i), "Main node");
    await user.click(screen.getByRole("button", { name: /continue/i }));
    await user.type(screen.getByLabelText(/ethernet ipv4 address/i), "10.10.10.2");
    await user.click(screen.getByRole("button", { name: /^connect$/i }));
    await waitFor(() => expect(connectPeer).toHaveBeenCalledWith("10.10.10.2"));
  });

  it("masks the API key and can reveal it", async () => {
    const user = userEvent.setup();
    render(<App service={serviceWith(readySnapshot)} />);
    await screen.findByText(/cluster overview/i);
    await user.click(screen.getByRole("button", { name: /api/i }));

    expect(await screen.findByText(/sk-local-••••••••••/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /show api key/i }));
    expect(screen.getByText("sk-local-1234567890")).toBeInTheDocument();
  });

  it("explains why chat is unavailable without a runtime or model", async () => {
    const user = userEvent.setup();
    const unavailable = {
      ...readySnapshot,
      runtime: { status: "missing" as const },
      models: [],
      cluster: { status: "idle" as const },
    };
    render(<App service={serviceWith(unavailable)} />);
    await screen.findByText(/cluster overview/i);
    await user.click(screen.getByRole("button", { name: /chat/i }));

    expect(screen.getByRole("alert")).toHaveTextContent(/runtime is not installed/i);
    expect(screen.getByRole("button", { name: /send message/i })).toBeDisabled();
  });
});
