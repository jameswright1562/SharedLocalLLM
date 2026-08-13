import type { PageProps } from "../types";
import { ComputePath } from "../components/ComputePath";
import { Meter, StatusPill } from "../components/Telemetry";
import { fitLabels, formatGb } from "./pageFormat";

export function OverviewPage({ snapshot, navigate }: PageProps) {
  const combinedVram = snapshot.nodes.reduce((sum, node) => sum + node.gpu.vramAvailableGb, 0);
  const combinedRam = snapshot.nodes.reduce((sum, node) => sum + node.ramAvailableGb, 0);
  const clusterModel = snapshot.models.find((model) => model.id === snapshot.cluster.modelId);

  return (
    <div className="page">
      <header className="page-header split-header">
        <div>
          <p className="section-kicker">Control plane</p>
          <h1>Cluster overview</h1>
          <p>Live capacity and routing across this trusted pair.</p>
        </div>
        <button className="button secondary" onClick={() => navigate("models")}>
          Choose model
        </button>
      </header>

      {snapshot.cluster.error && (
        <div className="error-panel" role="alert">
          <span className="status-dot danger" />
          <div>
            <strong>Cluster stopped</strong>
            <p>{snapshot.cluster.error}</p>
          </div>
        </div>
      )}

      <ComputePath cluster={snapshot.cluster} nodes={snapshot.nodes} />

      <section className="summary-strip" aria-label="Cluster summary">
        <div>
          <span>Usable GPU memory</span>
          <strong>{formatGb(combinedVram)}</strong>
          <small>
            Across {snapshot.nodes.length} active {snapshot.nodes.length === 1 ? "node" : "nodes"}
          </small>
        </div>
        <div>
          <span>Available system memory</span>
          <strong>{formatGb(combinedRam)}</strong>
          <small>Safety reserve already excluded</small>
        </div>
        <div>
          <span>Active model</span>
          <strong className={clusterModel ? "" : "muted-value"}>
            {clusterModel?.name ?? "None loaded"}
          </strong>
          <small>{clusterModel ? fitLabels[clusterModel.fit] : "Choose a model to begin"}</small>
        </div>
        <div>
          <span>Link quality</span>
          <strong className="capitalize">{snapshot.network?.classification ?? "Untested"}</strong>
          <small>
            {snapshot.network
              ? `${Math.round(Math.min(snapshot.network.downMbps, snapshot.network.upMbps))} Mbit/s · ${snapshot.network.latencyP95Ms} ms p95`
              : "Run link diagnostics"}
          </small>
        </div>
      </section>

      <div className="section-title">
        <div>
          <p className="section-kicker">Resource map</p>
          <h2>Compute nodes</h2>
        </div>
        <button className="text-button" onClick={() => navigate("nodes")}>
          Inspect hardware →
        </button>
      </div>
      <div className="node-grid">
        {snapshot.nodes.slice(0, 2).map((node) => (
          <article className="node-card" key={node.id}>
            <header>
              <div>
                <span className="role-label">{node.role}</span>
                <h3>{node.name}</h3>
              </div>
              <StatusPill online={node.online}>{node.online ? "Online" : "Offline"}</StatusPill>
            </header>
            <div className="hardware-name">
              <span>GPU</span>
              <strong>{node.gpu.name}</strong>
            </div>
            <div className="resource-row">
              <div>
                <span>Free VRAM</span>
                <strong>{formatGb(node.gpu.vramAvailableGb)}</strong>
              </div>
              <span>{formatGb(node.gpu.vramTotalGb)} total</span>
            </div>
            <Meter value={node.gpu.vramAvailableGb} max={node.gpu.vramTotalGb} />
            <div className="resource-row">
              <div>
                <span>Free RAM</span>
                <strong>{formatGb(node.ramAvailableGb)}</strong>
              </div>
              <span>{formatGb(node.ramTotalGb)} total</span>
            </div>
            <Meter value={node.ramAvailableGb} max={node.ramTotalGb} tone="amber" />
            <footer>
              <span>{node.adapter.name}</span>
              <span>
                {node.adapter.linkSpeedMbps
                  ? `${node.adapter.linkSpeedMbps} Mbit/s link`
                  : "Link speed unknown"}
              </span>
            </footer>
          </article>
        ))}
        {snapshot.nodes.length < 2 && (
          <article className="node-card empty-node">
            <span className="empty-node-symbol" aria-hidden="true">
              ＋
            </span>
            <h3>Pair a second computer</h3>
            <p>Add another node to combine GPU memory or move work to the faster machine.</p>
            <button className="button secondary" onClick={() => navigate("settings")}>
              Open pairing
            </button>
          </article>
        )}
      </div>
    </div>
  );
}
