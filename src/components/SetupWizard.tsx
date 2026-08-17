import { useEffect, useMemo, useState } from "react";

import { describeAppError } from "../services/errors";
import type { AppService, AppSnapshot, NetworkBenchmark, NodeCapabilities } from "../types";
import { SetupStepContent } from "./SetupStepContent";

interface SetupWizardProps {
  snapshot: AppSnapshot;
  service: AppService;
  onComplete: (snapshot: AppSnapshot) => void;
}

const steps = ["Runtime", "Device", "Pair", "Models", "Network", "Ready"];

export function SetupWizard({ snapshot, service, onComplete }: SetupWizardProps) {
  const [liveSnapshot, setLiveSnapshot] = useState(snapshot);
  const initialStep = snapshot.runtime.status === "ready" ? 1 : 0;
  const [step, setStep] = useState(initialStep);
  const [deviceName, setDeviceName] = useState(snapshot.deviceName || "Local node");
  const [pairCode, setPairCode] = useState("");
  const [manualEndpoint, setManualEndpoint] = useState("");
  const [generatedCode, setGeneratedCode] = useState("");
  const [pairedNode, setPairedNode] = useState<NodeCapabilities | null>(snapshot.nodes[1] ?? null);
  const [network, setNetwork] = useState<NetworkBenchmark | undefined>(snapshot.network);
  const [busy, setBusy] = useState(false);
  const [runtimeProgress, setRuntimeProgress] = useState({
    percent: 0,
    status: "Ready to download",
  });
  const [error, setError] = useState("");

  const progress = useMemo(() => ((step + 1) / steps.length) * 100, [step]);

  function clearError() {
    setError("");
  }

  function reportError(reason: unknown, fallback: string) {
    setError(describeAppError(reason, fallback));
  }

  useEffect(() => {
    if (step !== 2 || pairedNode) return;
    let active = true;
    const poll = async () => {
      try {
        const next = await service.getAppSnapshot();
        if (!active) return;
        setLiveSnapshot(next);
        if (next.nodes.length > 1) setPairedNode(next.nodes[1] ?? null);
      } catch {
        /* keep waiting for the code host to persist the peer */
      }
    };
    const timer = window.setInterval(() => void poll(), 2_000);
    void poll();
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [pairedNode, service, step]);

  async function checkAgain() {
    setBusy(true);
    clearError();
    try {
      const next = await service.refreshHardware();
      setLiveSnapshot(next);
      if (next.runtime.status === "ready") setStep(1);
    } catch (reason) {
      reportError(reason, "Hardware and runtime status could not be refreshed.");
    } finally {
      setBusy(false);
    }
  }

  async function createCode() {
    setBusy(true);
    clearError();
    try {
      const result = await service.generatePairingCode();
      setGeneratedCode(result.code);
    } catch (reason) {
      reportError(reason, "Could not create a pairing code.");
    } finally {
      setBusy(false);
    }
  }

  async function installRuntime() {
    setBusy(true);
    clearError();
    try {
      await service.installRuntime((percent, status) => setRuntimeProgress({ percent, status }));
      setRuntimeProgress({ percent: 100, status: "Runtime ready" });
      setStep(1);
    } catch (reason) {
      reportError(reason, "The runtime could not be installed.");
    } finally {
      setBusy(false);
    }
  }

  async function pair() {
    if (!pairCode.trim()) return;
    setBusy(true);
    clearError();
    try {
      const normalizedCode = pairCode.replace(/\s/g, "");
      const endpoint = manualEndpoint.trim();
      const node = endpoint
        ? await service.pairWithPeer(normalizedCode, endpoint)
        : await service.pairWithPeer(normalizedCode);
      setPairedNode(node);
      setStep(3);
    } catch (reason) {
      reportError(reason, "Pairing failed. Check the code and the connection.");
    } finally {
      setBusy(false);
    }
  }

  async function addFolder() {
    setBusy(true);
    clearError();
    try {
      const directory = await service.addModelDirectory();
      if (!directory) {
        setError("No folder selected. Use detected sources or choose another folder.");
        return;
      }
      setStep(4);
    } catch (reason) {
      reportError(reason, "The model folder could not be added.");
    } finally {
      setBusy(false);
    }
  }

  async function testNetwork() {
    setBusy(true);
    clearError();
    try {
      const result = await service.runNetworkTest();
      setNetwork(result);
      setStep(5);
    } catch (reason) {
      reportError(reason, "The network test could not finish.");
    } finally {
      setBusy(false);
    }
  }

  async function finish() {
    setBusy(true);
    clearError();
    try {
      onComplete(await service.completeSetup(deviceName.trim()));
    } catch (reason) {
      reportError(reason, "Setup could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="setup-shell">
      <aside className="setup-rail" aria-label="Setup progress">
        <div className="brand-mark" aria-hidden="true">
          <span />
          <span />
        </div>
        <p className="eyebrow">SharedLocalLLM</p>
        <h1>Build your compute link.</h1>
        <p>
          Connect two trusted Windows computers, inspect the link, then load one model across the
          memory they can safely share.
        </p>
        <ol>
          {steps.map((label, index) => (
            <li key={label} className={index === step ? "current" : index < step ? "complete" : ""}>
              <span>{index < step ? "✓" : String(index + 1).padStart(2, "0")}</span>
              {label}
            </li>
          ))}
        </ol>
        <div className="setup-progress" aria-hidden="true">
          <span style={{ width: `${progress}%` }} />
        </div>
      </aside>
      <main className="setup-panel">
        <div className="setup-step-count">
          Step {step + 1} of {steps.length}
        </div>
        <SetupStepContent
          step={step}
          snapshot={liveSnapshot}
          service={service}
          deviceName={deviceName}
          setDeviceName={setDeviceName}
          pairCode={pairCode}
          setPairCode={setPairCode}
          manualEndpoint={manualEndpoint}
          setManualEndpoint={setManualEndpoint}
          generatedCode={generatedCode}
          pairedNode={pairedNode}
          network={network}
          busy={busy}
          runtimeProgress={runtimeProgress}
          installRuntime={installRuntime}
          checkAgain={checkAgain}
          createCode={createCode}
          pair={pair}
          addFolder={addFolder}
          testNetwork={testNetwork}
          setStep={setStep}
          finish={finish}
        />
        {error && (
          <div className="form-error" role="alert">
            <p>{error}</p>
          </div>
        )}
      </main>
    </div>
  );
}
