import { useState } from "react";
import { PairingPanel } from "../components/PairingPanel";
import { describeAppError } from "../services/errors";
import type { PageProps } from "../types";
import { StatusPill } from "../components/Telemetry";
import { formatGb } from "./pageFormat";

export function NodesPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [refreshing, setRefreshing] = useState(false);
  const [confirmingReset, setConfirmingReset] = useState(false);
  const [message, setMessage] = useState("");
  const [manualEndpoint, setManualEndpoint] = useState("");
  async function refresh() {
    setRefreshing(true);
    setMessage("");
    try {
      await service.refreshHardware();
      await refreshSnapshot();
    } catch (reason) {
      setMessage(describeAppError(reason, "Hardware refresh failed."));
    } finally {
      setRefreshing(false);
    }
  }
  async function resetPairing() {
    setRefreshing(true);
    setMessage("");
    try {
      await service.resetPairing();
      await refreshSnapshot();
      setConfirmingReset(false);
      setMessage("Paired node forgotten. Connect again to rejoin the peer.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The paired node could not be forgotten."));
    } finally {
      setRefreshing(false);
    }
  }
  async function connect() {
    setRefreshing(true);
    setMessage("");
    try {
      const endpoint = manualEndpoint.trim();
      await (endpoint ? service.connectPeer(endpoint) : service.connectPeer());
      await refreshSnapshot();
      setMessage("Connected to the other computer.");
    } catch (reason) {
      setMessage(describeAppError(reason, "Could not connect to the other computer."));
    } finally {
      setRefreshing(false);
    }
  }
  return (
    <div className="page">
      <header className="page-header split-header">
        <div>
          <p className="section-kicker">Inventory</p>
          <h1>Node capabilities</h1>
          <p>Hardware and availability reported by each trusted computer.</p>
        </div>
        <button className="button secondary" disabled={refreshing} onClick={() => void refresh()}>
          {refreshing ? "Refreshing…" : "Refresh hardware"}
        </button>
      </header>
      <div className="detail-node-list">
        {snapshot.nodes.map((node, index) => (
          <article className="detail-node" key={node.id}>
            <div className="node-ordinal">NODE {String(index + 1).padStart(2, "0")}</div>
            <header>
              <div>
                <h2>{node.name}</h2>
                <p>
                  {node.role} · {node.cpu}
                </p>
              </div>
              <StatusPill online={node.online}>{node.online ? "Reachable" : "Offline"}</StatusPill>
            </header>
            <dl className="spec-grid">
              <div>
                <dt>Graphics processor</dt>
                <dd>{node.gpu.name}</dd>
              </div>
              <div>
                <dt>GPU memory</dt>
                <dd>
                  {formatGb(node.gpu.vramAvailableGb)} free / {formatGb(node.gpu.vramTotalGb)}
                </dd>
              </div>
              <div>
                <dt>System memory</dt>
                <dd>
                  {formatGb(node.ramAvailableGb)} free / {formatGb(node.ramTotalGb)}
                </dd>
              </div>
              <div>
                <dt>Network path</dt>
                <dd>
                  {node.adapter.name}
                  {node.adapter.linkSpeedMbps ? ` · ${node.adapter.linkSpeedMbps} Mbit/s` : ""}
                </dd>
              </div>
            </dl>
            {index > 0 && !confirmingReset && (
              <button
                className="text-button danger-text"
                disabled={refreshing}
                onClick={() => setConfirmingReset(true)}
              >
                Forget {node.name}
              </button>
            )}
            {index > 0 && confirmingReset && (
              <div className="error-panel" role="alert">
                <div>
                  <strong>Forget {node.name}?</strong>
                  <p>
                    Trust and connection settings will reset. Model files and folders stay
                    untouched.
                  </p>
                  <div className="button-row">
                    <button
                      className="button secondary"
                      disabled={refreshing}
                      onClick={() => setConfirmingReset(false)}
                    >
                      Keep node
                    </button>
                    <button
                      className="button stop-button"
                      aria-label={`Confirm forget ${node.name}`}
                      disabled={refreshing}
                      onClick={() => void resetPairing()}
                    >
                      {refreshing ? "Forgetting…" : `Forget ${node.name}`}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </article>
        ))}
        {snapshot.nodes.length < 2 && (
          <div className="empty-state compact">
            <span>02</span>
            <div>
              <h2>No worker connected</h2>
              <p>
                The local node can still run models that fit. Enter a peer IP or let discovery find
                the other computer.
              </p>
            </div>
          </div>
        )}
        {snapshot.nodes.length < 2 && (
          <PairingPanel
            manualEndpoint={manualEndpoint}
            setManualEndpoint={setManualEndpoint}
            pairedNode={null}
            busy={refreshing}
            connect={() => void connect()}
          />
        )}
      </div>
      {message && (
        <div className="toast-message" role="status">
          {message}
        </div>
      )}
    </div>
  );
}
