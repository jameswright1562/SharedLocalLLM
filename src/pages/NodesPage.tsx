import { useState } from "react";
import type { PageProps } from "../types";
import { StatusPill } from "../components/Telemetry";
import { formatGb } from "./pageFormat";

export function NodesPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [refreshing, setRefreshing] = useState(false);
  async function refresh() {
    setRefreshing(true);
    try {
      await service.refreshHardware();
      await refreshSnapshot();
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
          </article>
        ))}
        {snapshot.nodes.length < 2 && (
          <div className="empty-state compact">
            <span>02</span>
            <div>
              <h2>No worker paired</h2>
              <p>
                The local node can still run models that fit. Pair another computer from Settings to
                pool resources.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
