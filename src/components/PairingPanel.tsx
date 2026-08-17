import type { NodeCapabilities } from "../types";

interface PairingPanelProps {
  generatedCode: string;
  pairCode: string;
  setPairCode: (value: string) => void;
  manualEndpoint: string;
  setManualEndpoint: (value: string) => void;
  pairedNode: NodeCapabilities | null;
  busy: boolean;
  createCode: () => void;
  pair: () => void;
  onContinue?: () => void;
}

export function PairingPanel({
  generatedCode,
  pairCode,
  setPairCode,
  manualEndpoint,
  setManualEndpoint,
  pairedNode,
  busy,
  createCode,
  pair,
  onContinue,
}: PairingPanelProps) {
  return (
    <>
      <div className="pair-grid">
        <div className="pair-option">
          <span className="option-number">A</span>
          <h3>Show a code</h3>
          {generatedCode ? (
            <output className="pair-code" aria-live="polite">
              {generatedCode}
            </output>
          ) : (
            <p>Valid for five minutes.</p>
          )}
          <button className="button secondary" disabled={busy} onClick={createCode}>
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
            onClick={pair}
          >
            {busy
              ? manualEndpoint.trim()
                ? "Connecting over Ethernet…"
                : "Searching local network…"
              : "Pair computers"}
          </button>
        </div>
      </div>
      {pairedNode && (
        <p className="inline-success">
          <span>✓</span> Paired with {pairedNode.name}
        </p>
      )}
      {pairedNode && onContinue && (
        <div className="button-row">
          <button className="button primary" onClick={onContinue}>
            Continue
          </button>
        </div>
      )}
    </>
  );
}
