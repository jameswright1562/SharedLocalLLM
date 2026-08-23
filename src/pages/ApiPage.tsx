import { useEffect, useState } from "react";
import { describeAppError } from "../services/errors";
import type { ApiConfig, ApiTryResult, PageProps } from "../types";

function maskApiKey(key: string) {
  const prefix = key.startsWith("sk-local-") ? "sk-local-" : key.slice(0, 4);
  return `${prefix}••••••••••`;
}

function formatBody(body: string) {
  try {
    return JSON.stringify(JSON.parse(body), null, 2);
  } catch {
    return body;
  }
}

export function ApiPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [config, setConfig] = useState<ApiConfig | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState<"url" | "key" | "curl" | "">("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [trying, setTrying] = useState(false);
  const [tryResult, setTryResult] = useState<ApiTryResult | null>(null);
  useEffect(() => {
    let active = true;
    void service
      .getApiConfig()
      .then((nextConfig) => {
        if (active) setConfig(nextConfig);
      })
      .catch((reason) => {
        if (active) setError(describeAppError(reason, "Could not read the API configuration."));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [service]);
  async function copy(value: string, what: "url" | "key" | "curl") {
    setError("");
    try {
      if (!navigator.clipboard) throw new Error("Clipboard is unavailable in this environment.");
      await navigator.clipboard.writeText(value);
      setCopied(what);
      window.setTimeout(() => setCopied(""), 1400);
    } catch (reason) {
      setError(describeAppError(reason, "The value could not be copied."));
    }
  }
  async function regenerate() {
    if (snapshot.cluster.status === "running") {
      const confirmed = window.confirm(
        "Regenerating the API key stops the running cluster. Continue?",
      );
      if (!confirmed) return;
    }
    setBusy(true);
    setError("");
    try {
      setConfig(await service.regenerateApiKey());
      setRevealed(true);
      await refreshSnapshot();
    } catch (reason) {
      setError(describeAppError(reason, "The API key could not be regenerated."));
    } finally {
      setBusy(false);
    }
  }
  async function runExample() {
    setTrying(true);
    setTryResult(null);
    setError("");
    try {
      setTryResult(await service.tryApiRequest());
    } catch (reason) {
      setError(describeAppError(reason, "The example request could not be completed."));
    } finally {
      setTrying(false);
    }
  }
  const curlPayload = JSON.stringify({
    model: "local-model",
    messages: [{ role: "user", content: "Hello" }],
  });
  const authHeader = config?.authRequired ? ` -H "Authorization: Bearer ${config.apiKey}"` : "";
  const curl = config
    ? `curl.exe -X POST "${config.url}/v1/chat/completions"${authHeader} -H "Content-Type: application/json" --data-raw '${curlPayload}'`
    : "";
  return (
    <div className="page">
      <header className="page-header">
        <p className="section-kicker">Loopback interface</p>
        <h1>Local API</h1>
        <p>
          Connect local tools using an OpenAI-compatible endpoint. It is never exposed to the LAN.
        </p>
      </header>
      {error && (
        <div className="error-panel" role="alert">
          {error}
        </div>
      )}
      {loading ? (
        <div className="loading-panel" role="status" aria-live="polite">
          <span className="spinner" /> Reading API configuration…
        </div>
      ) : !config ? (
        <div className="empty-state">
          <span>!</span>
          <div>
            <h2>API configuration unavailable</h2>
            <p>Resolve the error above, then reopen this page.</p>
          </div>
        </div>
      ) : (
        <>
          <section className="api-status-panel">
            <div>
              <span className={`large-health ${config.healthy ? "healthy" : "unhealthy"}`}>
                <i aria-hidden="true" />
              </span>
              <div>
                <p className="section-kicker">Connection health</p>
                <h2>{config.healthy ? "Listening on loopback" : "API unavailable"}</h2>
                <p>
                  {config.healthy
                    ? "Requests from this computer can reach the active coordinator."
                    : "Start the API service or resolve the port conflict in Settings."}
                </p>
              </div>
            </div>
            <span className={`status-pill ${config.healthy ? "online" : "offline"}`}>
              <i aria-hidden="true" />
              {config.healthy ? "Healthy" : "Offline"}
            </span>
          </section>
          <section className="credential-panel">
            <div className="credential-row">
              <span className="credential-label">Base URL</span>
              <code>{config.url}</code>
              <button className="button secondary" onClick={() => void copy(config.url, "url")}>
                {copied === "url" ? "Copied" : "Copy"}
              </button>
            </div>
            <div className="credential-row">
              <span className="credential-label">API key</span>
              <code>{revealed ? config.apiKey : maskApiKey(config.apiKey)}</code>
              <button
                className="icon-button"
                aria-label={revealed ? "Hide API key" : "Show API key"}
                onClick={() => setRevealed(!revealed)}
              >
                {revealed ? "◉" : "○"}
              </button>
              <button className="button secondary" onClick={() => void copy(config.apiKey, "key")}>
                {copied === "key" ? "Copied" : "Copy"}
              </button>
            </div>
            <div className="credential-row">
              <span className="credential-label">Authentication</span>
              <code>{config.authRequired ? "Bearer key required" : "Disabled — open access"}</code>
            </div>
            <div className="key-actions">
              <p>
                {config.authRequired
                  ? "Keep this key private. Regenerating it immediately invalidates the previous key."
                  : "Key checks are off in Settings, so any local tool can call this API without a key."}
              </p>
              <button
                className="text-button danger-text"
                disabled={busy}
                onClick={() => void regenerate()}
              >
                {busy ? "Regenerating…" : "Regenerate key"}
              </button>
            </div>
          </section>
          <section className="code-example">
            <header>
              <div>
                <span className="code-language">PowerShell / curl</span>
                <h2>Chat completion</h2>
              </div>
              <div className="code-actions">
                <button
                  className="button primary"
                  disabled={trying}
                  onClick={() => void runExample()}
                >
                  {trying ? "Running…" : "Try it"}
                </button>
                <button className="button secondary" onClick={() => void copy(curl, "curl")}>
                  {copied === "curl" ? "Copied" : "Copy example"}
                </button>
              </div>
            </header>
            <pre>
              <code>{curl}</code>
            </pre>
            {tryResult && (
              <div aria-label="Example response" className="try-result">
                <div className="try-result-meta">
                  <span
                    className={`status-pill ${tryResult.status >= 200 && tryResult.status < 400 ? "online" : "offline"}`}
                  >
                    <i aria-hidden="true" />
                    HTTP {tryResult.status}
                  </span>
                  <small>{tryResult.durationMs} ms</small>
                </div>
                <pre>
                  <code>{formatBody(tryResult.body)}</code>
                </pre>
              </div>
            )}
          </section>
        </>
      )}
      <section className="endpoint-list">
        <h2>Supported endpoints</h2>
        <div>
          <code>GET</code>
          <strong>/health</strong>
          <span>Service and model readiness</span>
        </div>
        <div>
          <code>GET</code>
          <strong>/v1/models</strong>
          <span>Available local models</span>
        </div>
        <div>
          <code>POST</code>
          <strong>/v1/chat/completions</strong>
          <span>Chat, images, and streaming</span>
        </div>
        <div>
          <code>POST</code>
          <strong>/v1/completions</strong>
          <span>Text completions and streaming</span>
        </div>
      </section>
    </div>
  );
}
