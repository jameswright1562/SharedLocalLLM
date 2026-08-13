import type { Dispatch, SetStateAction } from "react";
import type { AppService, AppSnapshot, NetworkBenchmark, NodeCapabilities } from "../types";

interface SetupStepContentProps {
  step: number;
  snapshot: AppSnapshot;
  service: AppService;
  deviceName: string;
  setDeviceName: Dispatch<SetStateAction<string>>;
  pairCode: string;
  setPairCode: Dispatch<SetStateAction<string>>;
  manualEndpoint: string;
  setManualEndpoint: Dispatch<SetStateAction<string>>;
  generatedCode: string;
  pairedNode: NodeCapabilities | null;
  network?: NetworkBenchmark;
  busy: boolean;
  runtimeProgress: { percent: number; status: string };
  installRuntime: () => Promise<void>;
  createCode: () => Promise<void>;
  pair: () => Promise<void>;
  addFolder: () => Promise<void>;
  testNetwork: () => Promise<void>;
  finish: () => Promise<void>;
  setStep: Dispatch<SetStateAction<number>>;
}

export function SetupStepContent({
  step,
  snapshot,
  service,
  deviceName,
  setDeviceName,
  pairCode,
  setPairCode,
  manualEndpoint,
  setManualEndpoint,
  generatedCode,
  pairedNode,
  network,
  busy,
  runtimeProgress,
  installRuntime,
  createCode,
  pair,
  addFolder,
  testNetwork,
  finish,
  setStep,
}: SetupStepContentProps) {
  return (
    <>
      {step === 0 && (
        <section>
          <p className="section-kicker">Runtime readiness</p>
          <h2>Install the inference runtime</h2>
          <p className="lede">
            SharedLocalLLM uses a verified llama.cpp CUDA runtime. Development tools and LM Studio
            are not required.
          </p>
          <div className="readiness-check error-panel" role="status">
            <span className="status-dot warning" />
            <div>
              <strong>Runtime required</strong>
              <p>Complete the bundled runtime installation, then check again.</p>
            </div>
          </div>
          {busy && (
            <div className="install-progress" aria-live="polite">
              <div>
                <span>{runtimeProgress.status}</span>
                <strong>{runtimeProgress.percent}%</strong>
              </div>
              <span>
                <i style={{ width: `${runtimeProgress.percent}%` }} />
              </span>
            </div>
          )}
          <div className="button-row">
            <button
              className="button secondary"
              disabled={busy}
              onClick={() => void service.refreshHardware()}
            >
              Check again
            </button>
            <button
              className="button primary"
              disabled={busy}
              onClick={() => void installRuntime()}
            >
              {busy ? "Installing…" : "Install runtime"}
            </button>
          </div>
        </section>
      )}

      {step === 1 && (
        <section>
          <p className="section-kicker">Identity</p>
          <h2>Name this computer</h2>
          <p className="lede">
            Use a short name you will recognize when choosing a coordinator or reading benchmark
            results.
          </p>
          <label className="field-label" htmlFor="device-name">
            Device name
          </label>
          <input
            id="device-name"
            value={deviceName}
            onChange={(event) => setDeviceName(event.target.value)}
            autoFocus
          />
          <div className="detected-hardware">
            <span>Detected locally</span>
            <strong>{snapshot.nodes[0]?.gpu.name ?? "GPU scan pending"}</strong>
            <small>
              {snapshot.nodes[0]
                ? `${snapshot.nodes[0].ramTotalGb} GB system memory`
                : "Hardware will appear after refresh"}
            </small>
          </div>
          <div className="button-row">
            <button
              className="button primary"
              disabled={!deviceName.trim()}
              onClick={() => setStep(2)}
            >
              Continue
            </button>
          </div>
        </section>
      )}

      {step === 2 && (
        <section>
          <p className="section-kicker">Private peer channel</p>
          <h2>Pair the second computer</h2>
          <p className="lede">
            Open SharedLocalLLM on the other computer. Create a code on either screen and enter it
            on the other.
          </p>
          <div className="pair-grid">
            <div className="pair-option">
              <span className="option-number">A</span>
              <h3>Show a code</h3>
              {generatedCode ? (
                <output className="pair-code" aria-live="polite">
                  {generatedCode}
                </output>
              ) : (
                <p>Valid for five minutes on this private network.</p>
              )}
              <button
                className="button secondary"
                disabled={busy}
                onClick={() => void createCode()}
              >
                {generatedCode ? "Create new code" : "Create pairing code"}
              </button>
            </div>
            <div className="or-divider">or</div>
            <div className="pair-option">
              <span className="option-number">B</span>
              <h3>Enter their code</h3>
              <label className="field-label" htmlFor="pair-code">
                Enter code
              </label>
              <input
                id="pair-code"
                inputMode="numeric"
                placeholder="000 000"
                maxLength={7}
                value={pairCode}
                onChange={(event) => setPairCode(event.target.value)}
              />
              <label className="field-label" htmlFor="manual-peer-endpoint">
                Ethernet IPv4 address (optional)
              </label>
              <input
                id="manual-peer-endpoint"
                inputMode="decimal"
                placeholder="192.168.50.2"
                value={manualEndpoint}
                onChange={(event) => setManualEndpoint(event.target.value)}
              />
              <small>
                For a direct cable, enter the IPv4 address shown by <code>ipconfig</code> on the
                computer displaying the code. Port 49158 is automatic.
              </small>
              <button
                className="button primary"
                disabled={busy || pairCode.replace(/\s/g, "").length !== 6}
                onClick={() => void pair()}
              >
                {busy
                  ? manualEndpoint.trim()
                    ? "Connecting over Ethernet…"
                    : "Searching private LAN…"
                  : "Pair computers"}
              </button>
            </div>
          </div>
          {pairedNode && (
            <p className="inline-success">
              <span>✓</span> Paired with {pairedNode.name}
            </p>
          )}
        </section>
      )}

      {step === 3 && (
        <section>
          <p className="section-kicker">Model sources</p>
          <h2>Choose where models live</h2>
          <p className="lede">
            LM Studio folders are discovered when present. Add any other directory without moving or
            changing its files.
          </p>
          <div className="source-choice selected">
            <div>
              <span className="source-glyph">LM</span>
              <strong>LM Studio models</strong>
              <p>Automatic per-computer discovery</p>
            </div>
            <span className="tag cyan">Detected</span>
          </div>
          <button className="source-choice button-reset" onClick={() => void addFolder()}>
            <div>
              <span className="source-glyph">＋</span>
              <strong>Add a custom folder</strong>
              <p>Choose any directory containing GGUF files</p>
            </div>
            <span aria-hidden="true">→</span>
          </button>
          <div className="button-row">
            <button className="button secondary" onClick={() => setStep(4)}>
              Use detected sources
            </button>
          </div>
        </section>
      )}

      {step === 4 && (
        <section>
          <p className="section-kicker">Link test</p>
          <h2>Measure the path between nodes</h2>
          <p className="lede">
            Throughput and latency decide whether splitting a model will help. The test sends
            temporary encrypted data in both directions.
          </p>
          <div className="network-illustration" aria-hidden="true">
            <span>THIS PC</span>
            <i aria-hidden="true" />
            <b>↔</b>
            <i aria-hidden="true" />
            <span>PEER</span>
          </div>
          {network && (
            <p className="inline-success">
              <span>✓</span> Existing result:{" "}
              {Math.round(Math.min(network.downMbps, network.upMbps))} Mbit/s ·{" "}
              {network.latencyP95Ms} ms p95
            </p>
          )}
          <div className="button-row">
            <button className="button primary" disabled={busy} onClick={() => void testNetwork()}>
              {busy ? "Testing link…" : "Run network test"}
            </button>
          </div>
        </section>
      )}

      {step === 5 && (
        <section>
          <p className="section-kicker">Ready</p>
          <h2>Your compute link is ready</h2>
          <p className="lede">
            Models will be evaluated against available GPU memory, system memory, and the measured
            link before launch.
          </p>
          <div className="ready-summary">
            <div>
              <span>Nodes</span>
              <strong>{pairedNode || snapshot.nodes.length > 1 ? "2 online" : "1 local"}</strong>
            </div>
            <div>
              <span>Runtime</span>
              <strong>{snapshot.runtime.version ?? "Installed"}</strong>
            </div>
            <div>
              <span>Network</span>
              <strong className="capitalize">{network?.classification ?? "Not tested"}</strong>
            </div>
          </div>
          <div className="button-row">
            <button className="button primary" disabled={busy} onClick={() => void finish()}>
              {busy ? "Saving setup…" : "Open dashboard"}
            </button>
          </div>
        </section>
      )}
    </>
  );
}
