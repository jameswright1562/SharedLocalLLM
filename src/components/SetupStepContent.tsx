import type { Dispatch, SetStateAction } from "react";
import type { AppService, AppSnapshot, NetworkBenchmark, NodeCapabilities } from "../types";
import { PairingPanel } from "./PairingPanel";

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
  checkAgain: () => Promise<void>;
  createCode: () => Promise<void>;
  pair: () => Promise<void>;
  addFolder: () => Promise<void>;
  testNetwork: () => Promise<void>;
  finish: () => Promise<void>;
  setStep: Dispatch<SetStateAction<number>>;
}

export function SetupStepContent(props: SetupStepContentProps) {
  const { step } = props;
  return (
    <>
      {step === 0 && <RuntimeStep {...props} />}
      {step === 1 && <IdentityStep {...props} />}
      {step === 2 && <PairStep {...props} />}
      {step === 3 && <SourcesStep {...props} />}
      {step === 4 && <NetworkStep {...props} />}
      {step === 5 && <ReadyStep {...props} />}
    </>
  );
}

function RuntimeStep({
  snapshot,
  busy,
  runtimeProgress,
  installRuntime,
  checkAgain,
}: SetupStepContentProps) {
  const ready = snapshot.runtime.status === "ready";
  return (
    <section>
      <p className="section-kicker">Runtime readiness</p>
      <h2>Install the inference runtime</h2>
      <p className="lede">
        SharedLocalLLM uses a verified llama.cpp CUDA runtime. Development tools and LM Studio are
        not required.
      </p>
      <div className={`readiness-check ${ready ? "" : "error-panel"}`} role="status">
        <span className={`status-dot ${ready ? "ready" : "warning"}`} />
        <div>
          <strong>{ready ? "Runtime ready" : "Runtime required"}</strong>
          <p>
            {ready
              ? (snapshot.runtime.version ?? "The pinned runtime is installed.")
              : "Complete the bundled runtime installation, then check again."}
          </p>
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
        <button className="button secondary" disabled={busy} onClick={() => void checkAgain()}>
          Check again
        </button>
        <button className="button primary" disabled={busy} onClick={() => void installRuntime()}>
          {busy ? "Installing…" : ready ? "Reinstall runtime" : "Install runtime"}
        </button>
      </div>
    </section>
  );
}

function IdentityStep({ snapshot, deviceName, setDeviceName, setStep }: SetupStepContentProps) {
  const valid = deviceName.trim().length > 0 && deviceName.trim().length <= 80;
  return (
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
        maxLength={80}
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
        <button className="button secondary" onClick={() => setStep(0)}>
          Back
        </button>
        <button className="button primary" disabled={!valid} onClick={() => setStep(2)}>
          Continue
        </button>
      </div>
    </section>
  );
}

function PairStep({
  generatedCode,
  pairCode,
  setPairCode,
  manualEndpoint,
  setManualEndpoint,
  pairedNode,
  busy,
  createCode,
  pair,
  setStep,
}: SetupStepContentProps) {
  return (
    <section>
      <p className="section-kicker">Encrypted peer channel</p>
      <h2>Pair the second computer</h2>
      <p className="lede">
        Open SharedLocalLLM on the other computer. Create a code on either screen and enter it on
        the other. You can finish setup with one computer and pair later from Nodes.
      </p>
      <PairingPanel
        generatedCode={generatedCode}
        pairCode={pairCode}
        setPairCode={setPairCode}
        manualEndpoint={manualEndpoint}
        setManualEndpoint={setManualEndpoint}
        pairedNode={pairedNode}
        busy={busy}
        createCode={() => void createCode()}
        pair={() => void pair()}
        onContinue={() => setStep(3)}
      />
      <div className="button-row">
        <button className="button secondary" onClick={() => setStep(1)}>
          Back
        </button>
        <button className="button secondary" onClick={() => setStep(3)}>
          Skip and use this computer only
        </button>
      </div>
    </section>
  );
}

function SourcesStep({ snapshot, addFolder, setStep }: SetupStepContentProps) {
  const lmStudio = snapshot.modelDirectories.some((directory) => directory.source === "lm-studio");
  return (
    <section>
      <p className="section-kicker">Model sources</p>
      <h2>Choose where models live</h2>
      <p className="lede">
        LM Studio folders are discovered when present. Add any other directory without moving or
        changing its files. This computer indexes its own files only.
      </p>
      <div className={`source-choice ${lmStudio ? "selected" : ""}`}>
        <div>
          <span className="source-glyph">LM</span>
          <strong>LM Studio models</strong>
          <p>Automatic per-computer discovery</p>
        </div>
        <span className={`tag ${lmStudio ? "cyan" : ""}`}>
          {lmStudio ? "Detected" : "Not found"}
        </span>
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
        <button className="button secondary" onClick={() => setStep(2)}>
          Back
        </button>
        <button className="button secondary" onClick={() => setStep(4)}>
          Use detected sources
        </button>
      </div>
    </section>
  );
}

function NetworkStep({ network, busy, testNetwork, setStep }: SetupStepContentProps) {
  return (
    <section>
      <p className="section-kicker">Link test</p>
      <h2>Measure the path between nodes</h2>
      <p className="lede">
        The test sends temporary encrypted data over the peer channel. One computer can still run
        models that fit locally if you skip it.
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
          <span>✓</span> Existing result: {Math.round(network.downMbps)} Mbit/s ·{" "}
          {network.latencyP95Ms} ms p95
        </p>
      )}
      <div className="button-row">
        <button className="button secondary" onClick={() => setStep(3)}>
          Back
        </button>
        <button className="button secondary" onClick={() => setStep(5)}>
          Skip and use this computer only
        </button>
        <button className="button primary" disabled={busy} onClick={() => void testNetwork()}>
          {busy ? "Testing link…" : "Run network test"}
        </button>
      </div>
    </section>
  );
}

function ReadyStep({
  pairedNode,
  snapshot,
  network,
  busy,
  finish,
  setStep,
}: SetupStepContentProps) {
  return (
    <section>
      <p className="section-kicker">Ready</p>
      <h2>Your compute link is ready</h2>
      <p className="lede">
        Models will be evaluated against available GPU memory, system memory, and the measured link
        before launch.
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
        <button className="button secondary" onClick={() => setStep(4)}>
          Back
        </button>
        <button className="button primary" disabled={busy} onClick={() => void finish()}>
          {busy ? "Saving setup…" : "Open dashboard"}
        </button>
      </div>
    </section>
  );
}
