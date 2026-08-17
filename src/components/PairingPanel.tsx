import type { NodeCapabilities } from "../types";

interface PairingPanelProps {
  manualEndpoint: string;
  setManualEndpoint: (value: string) => void;
  pairedNode: NodeCapabilities | null;
  busy: boolean;
  connect: () => void;
  onContinue?: () => void;
}

export function PairingPanel({
  manualEndpoint,
  setManualEndpoint,
  pairedNode,
  busy,
  connect,
  onContinue,
}: PairingPanelProps) {
  return (
    <>
      <div className="pair-grid">
        <div className="pair-option">
          <span className="option-number">A</span>
          <h3>Connect to the second computer</h3>
          <p>
            Both computers auto-discover each other. For a direct cable, enter the peer's IPv4
            address.
          </p>
          <label className="field-label" htmlFor="manual-peer-endpoint">
            Ethernet IPv4 address (optional)
          </label>
          <input
            id="manual-peer-endpoint"
            inputMode="decimal"
            placeholder="10.10.10.2"
            value={manualEndpoint}
            onChange={(event) => setManualEndpoint(event.target.value)}
          />
          <small>Port 49158 is automatic.</small>
          <button className="button primary" disabled={busy} onClick={connect}>
            {busy ? "Connecting…" : "Connect"}
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
