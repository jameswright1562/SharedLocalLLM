import { useState } from "react";
import { describeAppError } from "../services/errors";
import type { PageProps } from "../types";
import { Switch } from "@mantine/core";

export function SettingsPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [tab, setTab] = useState<"general" | "runtime" | "sources" | "logs">("general");
  const [deviceName, setDeviceName] = useState(snapshot.deviceName);
  const [apiPort, setApiPort] = useState(snapshot.apiPort);
  const [authRequired, setAuthRequired] = useState(snapshot.authRequired);
  const [autostart, setAutostart] = useState(snapshot.autostart);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [runtimeProgress, setRuntimeProgress] = useState("");

  async function saveSettings() {
    setBusy(true);
    setMessage("");
    try {
      await service.updateSettings({
        deviceName: deviceName.trim(),
        apiPort,
        authRequired,
        autostart,
      });
      await refreshSnapshot();
      setMessage("Settings saved.");
    } catch (reason) {
      setMessage(describeAppError(reason, "Settings could not be saved."));
    } finally {
      setBusy(false);
    }
  }

  async function addFolder() {
    setBusy(true);
    setMessage("");
    try {
      const directory = await service.addModelDirectory();
      if (!directory) {
        setMessage("No folder selected.");
        return;
      }
      await refreshSnapshot();
      setMessage("Model folder added.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The folder could not be added."));
    } finally {
      setBusy(false);
    }
  }

  async function removeFolder(id: string) {
    setBusy(true);
    setMessage("");
    try {
      await service.removeModelDirectory(id);
      await refreshSnapshot();
      setMessage("Model folder removed.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The folder could not be removed."));
    } finally {
      setBusy(false);
    }
  }

  async function installRuntime() {
    setBusy(true);
    setMessage("");
    try {
      await service.installRuntime((_percent, status) => setRuntimeProgress(status));
      await refreshSnapshot();
      setMessage("Runtime installed.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The runtime could not be installed."));
    } finally {
      setBusy(false);
    }
  }

  async function openLogs() {
    setMessage("");
    try {
      await service.openLogsFolder();
    } catch (reason) {
      setMessage(describeAppError(reason, "The logs folder could not be opened."));
    }
  }

  const tabs = ["general", "runtime", "sources", "logs"] as const;
  return (
    <div className="page">
      <header className="page-header">
        <p className="section-kicker">Application</p>
        <h1>Settings & logs</h1>
        <p>Runtime, local model sources, and diagnostics for this computer.</p>
      </header>
      <div className="settings-layout">
        <nav className="settings-nav" role="tablist" aria-label="Settings sections">
          {tabs.map((value) => (
            <button
              key={value}
              id={`settings-tab-${value}`}
              role="tab"
              aria-selected={tab === value}
              aria-controls={`settings-panel-${value}`}
              className={tab === value ? "active" : ""}
              onClick={() => setTab(value)}
            >
              {value === "sources"
                ? "Model sources"
                : value === "runtime"
                  ? "Runtime"
                  : `${value[0]!.toUpperCase()}${value.slice(1)}`}
            </button>
          ))}
        </nav>
        <section
          className="settings-content"
          id={`settings-panel-${tab}`}
          role="tabpanel"
          aria-labelledby={`settings-tab-${tab}`}
        >
          {tab === "general" && (
            <>
              <h2>General</h2>
              <div className="settings-row">
                <div>
                  <strong>Device name</strong>
                  <p>Shown to the paired computer.</p>
                </div>
                <input
                  aria-label="Device name"
                  maxLength={80}
                  value={deviceName}
                  onChange={(event) => setDeviceName(event.target.value)}
                />
              </div>
              <div className="settings-row">
                <div>
                  <strong>Local API port</strong>
                  <p>Loopback only. SharedLocalLLM will not silently choose a new port.</p>
                </div>
                <input
                  aria-label="Local API port"
                  className="narrow-input"
                  type="number"
                  min={1024}
                  max={65535}
                  value={apiPort}
                  onChange={(event) => setApiPort(Number(event.target.value))}
                />
              </div>
              <div className="settings-row">
                <div>
                  <strong>Require API key</strong>
                  <p>When off, local tools can call the loopback API without the bearer key.</p>
                </div>
                <Switch
                  role="switch"
                  aria-label="Require API key"
                  aria-checked={authRequired}
                  onClick={() => setAuthRequired(!authRequired)}
                />
              </div>
              <div className="settings-row">
                <div>
                  <strong>Start with Windows</strong>
                  <p>Launch to the notification area.</p>
                </div>
                <Switch
                  role="switch"
                  aria-label="Start with Windows"
                  aria-checked={autostart}
                  checked={autostart}
                  onClick={(e) => setAutostart(e.currentTarget.checked)}
                />
              </div>
              <div className="button-row">
                <button
                  className="button primary"
                  disabled={
                    busy ||
                    !deviceName.trim() ||
                    deviceName.trim().length > 80 ||
                    apiPort < 1024 ||
                    apiPort > 65535
                  }
                  onClick={() => void saveSettings()}
                >
                  {busy ? "Saving…" : "Save changes"}
                </button>
              </div>
            </>
          )}
          {tab === "runtime" && (
            <>
              <h2>Runtime</h2>
              <p>
                Install or repair the pinned llama.cpp CUDA runtime. Failed downloads are never
                activated.
              </p>
              <div className="settings-row">
                <div>
                  <strong>llama.cpp runtime</strong>
                  <p>
                    {snapshot.runtime.error ||
                      runtimeProgress ||
                      "Verified backend used by this computer."}
                  </p>
                </div>
                <span
                  className={`status-pill ${snapshot.runtime.status === "ready" ? "online" : "offline"}`}
                >
                  <i aria-hidden="true" />
                  {snapshot.runtime.version ?? snapshot.runtime.status}
                </span>
              </div>
              <div className="button-row">
                <button
                  className="button primary"
                  disabled={busy}
                  onClick={() => void installRuntime()}
                >
                  {busy
                    ? "Installing…"
                    : snapshot.runtime.status === "ready"
                      ? "Repair runtime"
                      : "Install runtime"}
                </button>
              </div>
            </>
          )}
          {tab === "sources" && (
            <>
              <div className="section-title">
                <div>
                  <h2>Model sources</h2>
                  <p>Directories are read-only and configured per computer.</p>
                </div>
                <button className="button primary" disabled={busy} onClick={() => void addFolder()}>
                  Add folder
                </button>
              </div>
              <div className="directory-list">
                {snapshot.modelDirectories.map((directory) => (
                  <div key={directory.id}>
                    <span className="source-glyph">
                      {directory.source === "lm-studio" ? "LM" : "＋"}
                    </span>
                    <div>
                      <strong>{directory.path}</strong>
                      <small>
                        {directory.source === "lm-studio"
                          ? "LM Studio · automatic"
                          : "Custom folder"}
                      </small>
                    </div>
                    {directory.source === "custom" && (
                      <button
                        className="text-button danger-text"
                        disabled={busy}
                        onClick={() => void removeFolder(directory.id)}
                      >
                        Remove
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
          {tab === "logs" && (
            <>
              <div className="section-title">
                <div>
                  <h2>Live logs</h2>
                  <p>Secrets, prompt text, and personal path prefixes are redacted.</p>
                </div>
                <button className="button secondary" onClick={() => void openLogs()}>
                  Open logs folder
                </button>
              </div>
              <pre className="log-viewer" aria-label="Application logs">
                {snapshot.logs.length
                  ? snapshot.logs.join("\n")
                  : "No log entries in this session."}
              </pre>
            </>
          )}
        </section>
      </div>
      {message && (
        <div className="toast-message" role="status">
          {message}
        </div>
      )}
    </div>
  );
}
