import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { cloneSnapshot, serviceWith } from "../test/fixtures";
import { SetupWizard } from "./SetupWizard";

describe("SetupWizard", () => {
  it("installs a missing runtime and reports native progress", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.runtime = { status: "missing" };
    const installRuntime = vi.fn().mockImplementation(async (progress) => {
      progress?.(35, "Downloading runtime");
      progress?.(100, "Runtime ready");
      return { ...snapshot, runtime: { status: "ready", version: "b7000" } };
    });
    render(
      <SetupWizard
        snapshot={snapshot}
        service={serviceWith(snapshot, {
          installRuntime,
          refreshHardware: vi.fn().mockResolvedValue(snapshot),
        })}
        onComplete={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /install the inference runtime/i }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /check again/i }));
    await user.click(screen.getByRole("button", { name: /^install runtime$/i }));
    expect(await screen.findByRole("heading", { name: /name this computer/i })).toBeInTheDocument();
    expect(installRuntime).toHaveBeenCalledTimes(1);
  });

  it("shows a runtime installation error without leaving the step", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.runtime = { status: "error", error: "missing archive" };
    render(
      <SetupWizard
        snapshot={snapshot}
        service={serviceWith(snapshot, {
          installRuntime: vi.fn().mockRejectedValue(new Error("Archive verification failed")),
        })}
        onComplete={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /^install runtime$/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Archive verification failed");
  });

  it("creates a pairing code directly without a public-network prompt", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    const generatePairingCode = vi
      .fn()
      .mockResolvedValue({ code: "321 654", expiresInSeconds: 300 });
    render(
      <SetupWizard
        snapshot={snapshot}
        service={serviceWith(snapshot, { generatePairingCode })}
        onComplete={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /continue/i }));
    await user.click(screen.getByRole("button", { name: /create pairing code/i }));

    expect(await screen.findByText("321 654")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /use this public network/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /open windows network settings/i }),
    ).not.toBeInTheDocument();
    expect(generatePairingCode).toHaveBeenCalledTimes(1);
    expect(generatePairingCode).toHaveBeenCalledWith();
  });

  it("lets the user navigate back through every setup step", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.nodes = snapshot.nodes.slice(0, 1);
    render(
      <SetupWizard snapshot={snapshot} service={serviceWith(snapshot)} onComplete={vi.fn()} />,
    );

    // Step 1 -> 2
    await user.type(screen.getByLabelText(/device name/i), "Back tester");
    await user.click(screen.getByRole("button", { name: /continue/i }));
    expect(screen.getByRole("heading", { name: /pair the second computer/i })).toBeInTheDocument();

    // Step 2 -> 1
    await user.click(screen.getByRole("button", { name: /^back$/i }));
    expect(screen.getByRole("heading", { name: /name this computer/i })).toBeInTheDocument();

    // Step 1 -> 2 -> 3
    await user.click(screen.getByRole("button", { name: /continue/i }));
    await user.click(screen.getByRole("button", { name: /skip and use this computer only/i }));
    expect(screen.getByRole("heading", { name: /choose where models live/i })).toBeInTheDocument();

    // Step 3 -> 2
    await user.click(screen.getByRole("button", { name: /^back$/i }));
    expect(screen.getByRole("heading", { name: /pair the second computer/i })).toBeInTheDocument();

    // Step 2 -> 3 -> 4
    await user.click(screen.getByRole("button", { name: /skip and use this computer only/i }));
    await user.click(screen.getByRole("button", { name: /use detected sources/i }));
    expect(
      screen.getByRole("heading", { name: /measure the path between nodes/i }),
    ).toBeInTheDocument();

    // Step 4 -> 3
    await user.click(screen.getByRole("button", { name: /^back$/i }));
    expect(screen.getByRole("heading", { name: /choose where models live/i })).toBeInTheDocument();

    // Step 3 -> 4 -> 5
    await user.click(screen.getByRole("button", { name: /use detected sources/i }));
    await user.click(screen.getByRole("button", { name: /run network test/i }));
    expect(
      await screen.findByRole("heading", { name: /compute link is ready/i }),
    ).toBeInTheDocument();

    // Step 5 -> 4
    await user.click(screen.getByRole("button", { name: /^back$/i }));
    expect(
      screen.getByRole("heading", { name: /measure the path between nodes/i }),
    ).toBeInTheDocument();
  });

  it("handles code generation and pairing failures, then completes every setup step", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.nodes = snapshot.nodes.slice(0, 1);
    snapshot.network = undefined;
    const generatePairingCode = vi
      .fn()
      .mockRejectedValueOnce("offline")
      .mockResolvedValue({ code: "555 111", expiresInSeconds: 300 });
    const pairWithPeer = vi
      .fn()
      .mockRejectedValueOnce(new Error("Code expired"))
      .mockResolvedValue(cloneSnapshot().nodes[1]);
    const runNetworkTest = vi
      .fn()
      .mockRejectedValueOnce("network unavailable")
      .mockResolvedValue(cloneSnapshot().network);
    const onComplete = vi.fn();
    render(
      <SetupWizard
        snapshot={snapshot}
        service={serviceWith(snapshot, { generatePairingCode, pairWithPeer, runNetworkTest })}
        onComplete={onComplete}
      />,
    );

    await user.clear(screen.getByLabelText(/device name/i));
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    await user.type(screen.getByLabelText(/device name/i), "Compute lead");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await user.click(screen.getByRole("button", { name: /create pairing code/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("offline");
    await user.click(screen.getByRole("button", { name: /create pairing code/i }));
    expect(await screen.findByText("555 111")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /create new code/i }));

    await user.type(screen.getByLabelText(/enter code/i), "123456");
    await user.click(screen.getByRole("button", { name: /pair computers/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Code expired");
    await user.click(screen.getByRole("button", { name: /pair computers/i }));
    expect(
      await screen.findByRole("heading", { name: /choose where models live/i }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /use detected sources/i }));
    await user.click(screen.getByRole("button", { name: /run network test/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("network unavailable");
    await user.click(screen.getByRole("button", { name: /run network test/i }));
    expect(
      await screen.findByRole("heading", { name: /compute link is ready/i }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /open dashboard/i }));
    await waitFor(() =>
      expect(onComplete).toHaveBeenCalledWith(
        expect.objectContaining({ setupComplete: true, deviceName: "Compute lead" }),
      ),
    );
  });

  it("adds a custom source and exposes folder errors", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    const addModelDirectory = vi
      .fn()
      .mockRejectedValueOnce("dialog unavailable")
      .mockResolvedValue({ id: "new", nodeId: "node-a", path: "D:\\AI", source: "custom" });
    render(
      <SetupWizard
        snapshot={snapshot}
        service={serviceWith(snapshot, { addModelDirectory })}
        onComplete={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /continue/i }));
    await user.type(screen.getByLabelText(/enter code/i), "222333");
    await user.click(screen.getByRole("button", { name: /pair computers/i }));
    expect(
      await screen.findByRole("heading", { name: /choose where models live/i }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /add a custom folder/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("dialog unavailable");
    await user.click(screen.getByRole("button", { name: /add a custom folder/i }));
    await waitFor(() => expect(addModelDirectory).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("heading", { name: /measure the path/i })).toBeInTheDocument();
  });

  it("persists completion through the native setup command", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.nodes = snapshot.nodes.slice(0, 1);
    snapshot.network = undefined;
    const completed = { ...snapshot, setupComplete: true, deviceName: "Persistent node" };
    const completeSetup = vi.fn().mockResolvedValue(completed);
    const onComplete = vi.fn();
    render(
      <SetupWizard
        snapshot={snapshot}
        service={serviceWith(snapshot, { completeSetup })}
        onComplete={onComplete}
      />,
    );

    await user.clear(screen.getByLabelText(/device name/i));
    await user.type(screen.getByLabelText(/device name/i), "Persistent node");
    await user.click(screen.getByRole("button", { name: /continue/i }));
    await user.type(screen.getByLabelText(/enter code/i), "123456");
    await user.click(screen.getByRole("button", { name: /pair computers/i }));
    await screen.findByRole("heading", { name: /choose where models live/i });
    await user.click(screen.getByRole("button", { name: /use detected sources/i }));
    await user.click(screen.getByRole("button", { name: /run network test/i }));
    await user.click(await screen.findByRole("button", { name: /open dashboard/i }));

    expect(completeSetup).toHaveBeenCalledWith("Persistent node");
    expect(onComplete).toHaveBeenCalledWith(completed);
  });
});
