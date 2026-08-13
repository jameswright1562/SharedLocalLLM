import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { cloneSnapshot, serviceWith } from "../test/fixtures";
import type { AppSnapshot, PageProps } from "../types";
import { ApiPage } from "./ApiPage";
import { ChatPage } from "./ChatPage";
import { SettingsPage } from "./SettingsPage";

function props(snapshot: AppSnapshot = cloneSnapshot(), serviceOverrides = {}): PageProps {
  return {
    snapshot,
    service: serviceWith(snapshot, serviceOverrides),
    refreshSnapshot: vi.fn().mockResolvedValue(undefined),
    navigate: vi.fn(),
  };
}

describe("interactive pages", () => {
  it("does not return the smooth-scroll result as an effect cleanup", async () => {
    const scrollResult = Promise.resolve();
    const scrollIntoView = vi
      .spyOn(HTMLElement.prototype, "scrollIntoView")
      .mockImplementation(() => scrollResult as unknown as void);
    const snapshot = cloneSnapshot();
    snapshot.cluster = { ...snapshot.cluster, status: "running", modelId: "model-text" };

    const view = render(<ChatPage {...props(snapshot)} />);
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalled());
    expect(() => view.unmount()).not.toThrow();

    scrollIntoView.mockRestore();
  });

  it("sends chat, edits settings, attaches images, and cancels generation", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.cluster = { ...snapshot.cluster, status: "running", modelId: "model-vision" };
    let resolveChat!: (value: { content: string }) => void;
    const sendChatMessage = vi
      .fn()
      .mockImplementationOnce(() => new Promise((resolve) => (resolveChat = resolve)))
      .mockResolvedValueOnce({ content: "Second answer" });
    const cancelGeneration = vi.fn().mockResolvedValue(undefined);
    render(<ChatPage {...props(snapshot, { sendChatMessage, cancelGeneration })} />);

    await user.click(screen.getByRole("button", { name: /generation settings/i }));
    await user.clear(screen.getByLabelText(/system prompt/i));
    await user.type(screen.getByLabelText(/system prompt/i), "Be exact");
    await user.click(screen.getByRole("button", { name: /close generation settings/i }));

    const file = new File(["image"], "diagram.png", { type: "image/png" });
    await user.upload(screen.getByLabelText(/attach image/i), file);
    expect(screen.getByText("diagram.png")).toBeInTheDocument();
    await user.type(screen.getByLabelText(/^message$/i), "Explain this route");
    await user.click(screen.getByRole("button", { name: /send message/i }));
    expect(screen.getByText(/generating/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /■ stop/i }));
    expect(cancelGeneration).toHaveBeenCalled();
    const firstCall = sendChatMessage.mock.calls[0]!;
    expect(firstCall[2]).toHaveLength(1);
    expect(firstCall[2][0]).toMatch(/^data:image\/png;base64,/);
    resolveChat({ content: "Route explained" });
    await Promise.resolve();
    expect(screen.queryByText("Route explained")).not.toBeInTheDocument();

    await user.type(screen.getByLabelText(/^message$/i), "Again");
    await user.keyboard("{Enter}");
    expect(await screen.findByText("Second answer")).toBeInTheDocument();
  });

  it("shows generation errors and retries the last user prompt", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.cluster = { ...snapshot.cluster, status: "running", modelId: "model-text" };
    const sendChatMessage = vi
      .fn()
      .mockRejectedValueOnce("backend stopped")
      .mockResolvedValueOnce({ content: "Recovered" });
    render(<ChatPage {...props(snapshot, { sendChatMessage })} />);
    await user.type(screen.getByLabelText(/^message$/i), "Retry me");
    await user.click(screen.getByRole("button", { name: /send message/i }));
    expect(await screen.findByText(/backend stopped/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /retry/i }));
    expect(await screen.findByText("Recovered")).toBeInTheDocument();
  });

  it("routes each unavailable chat state to the corrective page", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.models = [];
    snapshot.cluster = { status: "idle" };
    const pageProps = props(snapshot);
    const { rerender } = render(<ChatPage {...pageProps} />);
    await user.click(screen.getByRole("button", { name: /choose a model/i }));
    expect(pageProps.navigate).toHaveBeenCalledWith("models");

    snapshot.runtime = { status: "missing" };
    const missingRuntimeProps = props(snapshot);
    rerender(<ChatPage {...missingRuntimeProps} />);
    await user.click(screen.getByRole("button", { name: /open runtime settings/i }));
    expect(missingRuntimeProps.navigate).toHaveBeenCalledWith("settings");

    const loadedButStopped = cloneSnapshot();
    loadedButStopped.cluster = { status: "ready", modelId: "model-text" };
    rerender(<ChatPage {...props(loadedButStopped)} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/no model is loaded/i);
    expect(screen.getByRole("button", { name: /send message/i })).toBeDisabled();
  });

  it("reveals, copies, regenerates, and displays an unhealthy API", async () => {
    const user = userEvent.setup();
    const getApiConfig = vi.fn().mockResolvedValue({
      url: "http://127.0.0.1:11435",
      apiKey: "abcd-secret",
      healthy: false,
    });
    const regenerateApiKey = vi.fn().mockResolvedValue({
      url: "http://127.0.0.1:11435",
      apiKey: "sk-local-newkey",
      healthy: true,
    });
    render(<ApiPage {...props(cloneSnapshot(), { getApiConfig, regenerateApiKey })} />);
    expect(screen.getByText(/reading api configuration/i)).toHaveAttribute("role", "status");
    expect(await screen.findByText(/api unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/abcd••/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /show api key/i }));
    expect(screen.getByText("abcd-secret")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /hide api key/i }));
    await user.click(screen.getAllByRole("button", { name: /^copy$/i })[0]!);
    await user.click(screen.getByRole("button", { name: /copy example/i }));
    await user.click(screen.getByRole("button", { name: /regenerate key/i }));
    expect(await screen.findByText("sk-local-newkey")).toBeInTheDocument();
    const example = screen.getByText(/curl\.exe .*chat\/completions/i);
    expect(example.textContent).not.toContain("\n");
    expect(example.textContent).toContain("--data-raw");
  });

  it("shows API load and clipboard failures as actionable errors", async () => {
    const user = userEvent.setup();
    const getApiConfig = vi.fn().mockRejectedValue({
      code: "api_unavailable",
      message: "The API configuration is unavailable.",
      action: "Restart the local service.",
    });
    const pageProps = props(cloneSnapshot(), { getApiConfig });
    const { rerender } = render(<ApiPage {...pageProps} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/restart the local service/i);

    const originalClipboard = navigator.clipboard;
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined });
    rerender(<ApiPage key="healthy" {...props()} />);
    await screen.findByText(/listening on loopback/i);
    await user.click(screen.getAllByRole("button", { name: /^copy$/i })[0]!);
    expect(await screen.findByRole("alert")).toHaveTextContent(/clipboard is unavailable/i);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: originalClipboard });
  });

  it("installs the runtime from settings", async () => {
    const user = userEvent.setup();
    const installRuntime = vi.fn().mockResolvedValue(cloneSnapshot());
    const pageProps = props(cloneSnapshot(), { installRuntime });
    render(<SettingsPage {...pageProps} />);
    await user.click(screen.getByRole("tab", { name: /runtime/i }));
    await user.click(screen.getByRole("button", { name: /repair runtime/i }));
    expect(installRuntime).toHaveBeenCalled();
    await waitFor(() => expect(pageProps.refreshSnapshot).toHaveBeenCalled());
  });

  it("manages model sources and opens logs", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.modelDirectories.unshift({
      id: "lm",
      nodeId: "node-a",
      path: "C:\\Models",
      source: "lm-studio",
    });
    const addModelDirectory = vi.fn().mockResolvedValue(snapshot.modelDirectories[0]);
    const removeModelDirectory = vi.fn().mockResolvedValue(undefined);
    const openLogsFolder = vi.fn().mockResolvedValue(undefined);
    const pageProps = props(snapshot, {
      addModelDirectory,
      removeModelDirectory,
      openLogsFolder,
    });
    render(<SettingsPage {...pageProps} />);
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
    await user.click(screen.getByRole("tab", { name: /model sources/i }));
    await user.click(screen.getByRole("button", { name: /^add folder$/i }));
    await user.click(screen.getByRole("button", { name: /^remove$/i }));
    expect(addModelDirectory).toHaveBeenCalled();
    expect(removeModelDirectory).toHaveBeenCalledWith("dir-1");
    await waitFor(() => expect(pageProps.refreshSnapshot).toHaveBeenCalledTimes(2));
    await user.click(screen.getByRole("tab", { name: /^logs$/i }));
    expect(screen.getByLabelText(/application logs/i)).toHaveTextContent("Peer channel ready");
    await user.click(screen.getByRole("button", { name: /open logs folder/i }));
    expect(openLogsFolder).toHaveBeenCalled();
  });

  it("persists controlled settings and exposes tab semantics", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    snapshot.apiPort = 12000;
    snapshot.autostart = true;
    const updateSettings = vi.fn().mockResolvedValue({
      ...snapshot,
      deviceName: "Saved node",
      apiPort: 12001,
      autostart: false,
    });
    const pageProps = props(snapshot, { updateSettings });
    render(<SettingsPage {...pageProps} />);

    expect(screen.getByRole("tablist", { name: /settings sections/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /general/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
    await user.clear(screen.getByLabelText(/device name/i));
    await user.type(screen.getByLabelText(/device name/i), "Saved node");
    await user.clear(screen.getByLabelText(/local api port/i));
    await user.type(screen.getByLabelText(/local api port/i), "12001");
    await user.click(screen.getByRole("switch"));
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    expect(updateSettings).toHaveBeenCalledWith({
      deviceName: "Saved node",
      apiPort: 12001,
      autostart: false,
    });
    expect(pageProps.refreshSnapshot).toHaveBeenCalled();
  });

  it("keeps model sources unchanged when the folder dialog is cancelled", async () => {
    const user = userEvent.setup();
    const pageProps = props(cloneSnapshot(), {
      addModelDirectory: vi.fn().mockResolvedValue(null),
    });
    render(<SettingsPage {...pageProps} />);

    await user.click(screen.getByRole("tab", { name: /model sources/i }));
    await user.click(screen.getByRole("button", { name: /^add folder$/i }));
    expect(await screen.findByRole("status")).toHaveTextContent(/no folder selected/i);
    expect(pageProps.refreshSnapshot).not.toHaveBeenCalled();
  });
});
