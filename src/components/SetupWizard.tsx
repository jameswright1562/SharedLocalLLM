import { useMemo, useState } from "react";

import { decodeAppError, describeAppError } from "../services/errors";
import type { AppService, AppSnapshot, NetworkBenchmark, NodeCapabilities } from "../types";
import { SetupStepContent } from "./SetupStepContent";

interface SetupWizardProps {
  snapshot: AppSnapshot;
  service: AppService;
  onComplete: (snapshot: AppSnapshot) => void;
}

type PublicNetworkRetry = "create-code" | "pair";

const steps = ["Runtime", "Device", "Pair", "Models", "Network", "Ready"];

export function SetupWizard({ snapshot, service, onComplete }: SetupWizardProps) {
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
  const [errorCode, setErrorCode] = useState<string>();
  const [publicNetworkRetry, setPublicNetworkRetry] = useState<PublicNetworkRetry>();

  const progress = useMemo(() => ((step + 1) / steps.length) * 100, [step]);

  function clearError() {
    setError("");
    setErrorCode(undefined);
    setPublicNetworkRetry(undefined);
  }

  function reportError(reason: unknown, fallback: string) {
    setError(describeAppError(reason, fallback));
    setErrorCode(decodeAppError(reason).code);
  }

  async function createCode(allowPublicNetwork = false) {
    setBusy(true);
    clearError();
    try {
      const result = await service.generatePairingCode(allowPublicNetwork);
      setGeneratedCode(result.code);
    } catch (reason) {
      reportError(reason, "Could not create a pairing code.");
      if (decodeAppError(reason).code === "private_network_required") {
        setPublicNetworkRetry("create-code");
      }
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

  async function pair(allowPublicNetwork = false) {
    if (!pairCode.trim()) return;
    setBusy(true);
    clearError();
    try {
      const normalizedCode = pairCode.replace(/\s/g, "");
      const endpoint = manualEndpoint.trim();
      const node = endpoint
        ? await service.pairWithPeer(normalizedCode, allowPublicNetwork, endpoint)
        : await service.pairWithPeer(normalizedCode, allowPublicNetwork);
      setPairedNode(node);
      setStep(3);
    } catch (reason) {
      reportError(reason, "Pairing failed. Check the code and private network.");
      if (decodeAppError(reason).code === "private_network_required") {
        setPublicNetworkRetry("pair");
      }
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

  async function openNetworkSettings() {
    try {
      await service.openNetworkSettings();
    } catch (reason) {
      reportError(reason, "Windows network settings could not be opened.");
    }
  }

  function retryOnPublicNetwork() {
    if (publicNetworkRetry === "create-code") {
      void createCode(true);
    } else if (publicNetworkRetry === "pair") {
      void pair(true);
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
          snapshot={snapshot}
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
            {step === 2 && errorCode === "private_network_required" && (
              <div className="network-profile-help">
                <p>
                  Private means a home or work network you trust—not a private internet connection.
                  Public networks can contain devices you do not trust, so pairing is blocked by
                  default. The override permits a temporary five-minute pairing session; cluster
                  launch remains blocked on a Public profile. Showing a code may require Windows
                  approval for a temporary firewall rule.
                </p>
                <div className="network-profile-actions">
                  <button
                    className="button secondary compact-button"
                    onClick={() => void openNetworkSettings()}
                  >
                    Open Windows network settings
                  </button>
                  <button className="text-button" onClick={retryOnPublicNetwork}>
                    Use this public network
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
