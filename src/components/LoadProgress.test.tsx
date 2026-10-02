import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";

import { render } from "../test/render";
import { LoadProgress } from "./LoadProgress";
import type { ClusterSession } from "../types";

function loadingCluster(overrides: Partial<ClusterSession> = {}): ClusterSession {
  return { status: "loading", modelId: "model", ...overrides };
}

describe("LoadProgress", () => {
  it("renders nothing once loading has finished", () => {
    const { container } = render(
      <LoadProgress cluster={{ status: "running", modelId: "model" }} />,
    );
    expect(container.querySelector('[data-testid="load-progress"]')).toBeNull();
  });

  it("walks through named stages with the current one highlighted", () => {
    render(
      <LoadProgress
        cluster={loadingCluster({ stage: "staging", workerNodeId: "peer" })}
        workerName="DESKTOP"
      />,
    );

    expect(screen.getByTestId("load-progress")).toHaveTextContent(/staging worker devices/i);
    expect(screen.getByText("Preparing allocation")).toBeInTheDocument();
    expect(screen.getByText("Loading weights")).toBeInTheDocument();
    expect(screen.queryByText("Health check")).not.toBeInTheDocument();
  });

  it("shows the approximate transfer meter once bytes are reported", () => {
    render(
      <LoadProgress
        cluster={loadingCluster({
          stage: "loading_weights",
          bytesToWorker: 1_500_000_000,
          expectedBytes: 22_000_000_000,
        })}
        workerName="DESKTOP"
      />,
    );

    expect(screen.getByLabelText("Transfer progress")).toBeInTheDocument();
    expect(screen.getByTestId("load-progress")).toHaveTextContent(/sent to DESKTOP/);
    expect(screen.getByTestId("load-progress")).toHaveTextContent(/estimate/);
  });

  it("falls back to an indeterminate bar before counters arrive", () => {
    render(<LoadProgress cluster={loadingCluster({})} />);

    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
    expect(screen.getByTestId("load-progress")).toHaveTextContent(/starting/i);
    expect(screen.queryByText(/worker tunnel/i)).not.toBeInTheDocument();
  });

  it("shows only local compute stages for a local built-in load", () => {
    render(
      <LoadProgress
        cluster={loadingCluster({ stage: "staging", engine: "builtin", expectedBytes: 0 })}
      />,
    );
    expect(screen.getByText("Preparing compute devices")).toBeInTheDocument();
    expect(
      screen.queryByText(/worker tunnel|worker devices|transferring weights/i),
    ).not.toBeInTheDocument();
  });

  it("describes the server readiness wait without inventing a separate health stage", () => {
    render(
      <LoadProgress
        cluster={loadingCluster({
          stage: "loading_weights",
          engine: "llama-server",
          workerNodeId: "peer",
        })}
      />,
    );
    expect(screen.getByTestId("load-progress")).toHaveTextContent(
      /loading weights and checking readiness/i,
    );
    expect(screen.queryByText(/staging worker devices|health check/i)).not.toBeInTheDocument();
  });
});
