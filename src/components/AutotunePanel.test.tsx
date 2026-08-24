import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { render } from "../test/render";
import { cloneSnapshot, serviceWith } from "../test/fixtures";
import type { AutotuneStatus, AppService } from "../types";
import { AutotunePanel } from "./AutotunePanel";

const complete: AutotuneStatus = {
  status: "complete",
  modelId: "model-text",
  modelName: "Orchid 9B Q4_K_M",
  depth: "quick",
  stageIndex: 4,
  stageCount: 4,
  currentStage: null,
  result: {
    modelId: "model-text",
    modelName: "Orchid 9B Q4_K_M",
    depth: "quick",
    ranAt: "2026-08-20T10:00:00Z",
    fingerprint: "abc123",
    winners: {
      batchSize: 2048,
      uBatch: 512,
      cpuThreads: 8,
      kvCacheK: "q4_0",
      kvCacheV: "q8_0",
      gpuLayers: 32,
    },
    promptTokensPerSecond: 118.6,
    generationTokensPerSecond: 24.1,
  },
};

function setup(overrides: Partial<AppService> = {}) {
  const snapshot = cloneSnapshot();
  const model = snapshot.models[0]!;
  const onMessage = vi.fn();
  const onRefresh = vi.fn().mockResolvedValue(undefined);
  const onApplied = vi.fn();
  const service = serviceWith(snapshot, {
    startModelAutotune: vi.fn().mockResolvedValue(complete),
    getAutotuneStatus: vi.fn().mockResolvedValue(complete),
    cancelModelAutotune: vi.fn().mockResolvedValue(undefined),
    applyModelTune: vi.fn().mockResolvedValue({
      loadConfig: { contextSize: 4096, gpuLayers: [], batchSize: 2048 },
      staleTopology: false,
      tunedAt: complete.result?.ranAt,
    }),
    ...overrides,
  });
  render(
    <AutotunePanel
      model={model}
      tune={undefined}
      service={service}
      onMessage={onMessage}
      onRefresh={onRefresh}
      onApplied={onApplied}
    />,
  );
  return { service, onMessage, onRefresh, onApplied, model };
}

describe("AutotunePanel", () => {
  it("explains that a model is required", () => {
    render(
      <AutotunePanel
        service={serviceWith()}
        onMessage={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(screen.getByText(/select a model to tune/i)).toBeInTheDocument();
  });

  it("starts a run and shows the winning configuration", async () => {
    const user = userEvent.setup();
    const { service } = setup();

    await user.click(screen.getByRole("button", { name: /start tuning/i }));

    expect(service.startModelAutotune).toHaveBeenCalledWith(expect.any(String), "quick");
    await waitFor(() => expect(screen.getByTestId("autotune-winners")).toBeInTheDocument());
    expect(screen.getAllByText(/2048/).length).toBeGreaterThan(0);
  });

  it("applies the tuned settings and refreshes", async () => {
    const user = userEvent.setup();
    const { service, onMessage, onRefresh, onApplied } = setup();

    await user.click(screen.getByRole("button", { name: /start tuning/i }));
    await user.click(await screen.findByRole("button", { name: /apply tuned settings/i }));

    await waitFor(() => expect(service.applyModelTune).toHaveBeenCalledWith("model-text"));
    await waitFor(() => expect(onRefresh).toHaveBeenCalled());
    expect(onApplied).toHaveBeenCalledWith({ contextSize: 4096, gpuLayers: [], batchSize: 2048 });
    expect(onMessage.mock.calls.some(([text]) => /saved for/i.test(text as string))).toBe(true);
  });

  it("reports a failed run without offering winners", async () => {
    const user = userEvent.setup();
    setup({
      startModelAutotune: vi.fn().mockRejectedValue(new Error("llama-bench exited with code 1.")),
    });

    await user.click(screen.getByRole("button", { name: /start tuning/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/llama-bench exited/i);
    expect(screen.queryByTestId("autotune-winners")).not.toBeInTheDocument();
  });

  it("offers a stop control while a run is in progress", async () => {
    const user = userEvent.setup();
    const { service } = setup({
      startModelAutotune: vi.fn().mockResolvedValue({
        status: "running",
        modelId: "model-text",
        stageIndex: 2,
        stageCount: 4,
        currentStage: "GPU layer total",
      }),
    });

    await user.click(screen.getByRole("button", { name: /start tuning/i }));

    expect(await screen.findByText(/stage 2 of 4/i)).toBeInTheDocument();
    expect(screen.getByText(/GPU layer total/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /stop tuning/i }));
    expect(service.cancelModelAutotune).toHaveBeenCalled();
  });
});
