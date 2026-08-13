import type { ClusterSession, NodeCapabilities } from "../types";

interface ComputePathProps {
  cluster: ClusterSession;
  nodes: NodeCapabilities[];
  compact?: boolean;
}

function nodeName(nodes: NodeCapabilities[], id?: string, fallback = "Waiting for peer") {
  return nodes.find((node) => node.id === id)?.name ?? fallback;
}

export function ComputePath({ cluster, nodes, compact = false }: ComputePathProps) {
  const active = cluster.status === "loading" || cluster.status === "running";
  const coordinator = nodeName(nodes, cluster.coordinatorNodeId, "Choose coordinator");
  const worker = nodeName(nodes, cluster.workerNodeId);

  return (
    <section
      className={`compute-path ${active ? "is-active" : ""} ${compact ? "is-compact" : ""}`}
      aria-label="Compute path"
      aria-live="polite"
      data-testid="compute-path"
    >
      <div className="path-node">
        <span className="path-index">01</span>
        <span className="path-label">Local API</span>
        <strong>127.0.0.1</strong>
      </div>
      <div className="path-line" aria-hidden="true">
        <i />
      </div>
      <div className="path-node">
        <span className="path-index">02</span>
        <span className="path-label">Coordinator</span>
        <strong>{coordinator}</strong>
      </div>
      <div className="path-line" aria-hidden="true">
        <i />
      </div>
      <div className={`path-node ${nodes.length < 2 ? "path-node-empty" : ""}`}>
        <span className="path-index">03</span>
        <span className="path-label">Worker</span>
        <strong>{worker}</strong>
      </div>
      <span className={`path-state status-${cluster.status}`}>
        <i aria-hidden="true" /> {cluster.status}
      </span>
    </section>
  );
}
