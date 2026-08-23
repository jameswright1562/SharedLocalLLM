import { useState } from "react";
import { describeAppError } from "../services/errors";
import type { NetworkBenchmark, PageProps } from "../types";
import { Meter } from "../components/Telemetry";

export function NetworkPage({ snapshot, service }: PageProps) {
  const [result, setResult] = useState<NetworkBenchmark | undefined>(snapshot.network);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState("");

  async function runTest() {
    setTesting(true);
    setError("");
    try {
      setResult(await service.runNetworkTest());
    } catch (reason) {
      setError(describeAppError(reason, "Network test failed."));
    } finally {
      setTesting(false);
    }
  }

  const slowest = result ? Math.min(result.downMbps, result.upMbps) : 0;
  return (
    <div className="page">
      <header className="page-header split-header">
        <div>
          <p className="section-kicker">Transport</p>
          <h1>Link diagnostics</h1>
          <p>Measure the peer route used for distributed inference.</p>
        </div>
        <button
          className="button primary"
          disabled={testing || snapshot.nodes.length < 2}
          onClick={() => void runTest()}
        >
          {testing ? "Testing the peer channel…" : "Run network test"}
        </button>
      </header>
      {error && (
        <div className="error-panel" role="alert">
          <span className="status-dot danger" aria-hidden="true" />
          <div>
            <strong>Test failed</strong>
            <p>{error}</p>
          </div>
        </div>
      )}
      {!result ? (
        <div className="empty-state">
          <span>↔</span>
          <div>
            <h2>No link result yet</h2>
            <p>
              Pair a worker, close large transfers, and run the test to get a topology
              recommendation.
            </p>
          </div>
        </div>
      ) : (
        <div data-testid="network-test-result">
          <section className={`network-verdict verdict-${result.classification}`}>
            <div>
              <span className="signal-bars" aria-hidden="true">
                <i />
                <i />
                <i />
              </span>
              <div>
                <p className="section-kicker">Link classification</p>
                <strong className="classification capitalize">{result.classification}</strong>
              </div>
            </div>
            <p>
              {result.classification === "good"
                ? "This path is well suited to layer-split inference."
                : result.classification === "usable"
                  ? "Distributed inference should work, but compare it with single-node placement."
                  : "Prefer single-node inference when the model fits. The link may constrain token speed."}
            </p>
          </section>
          <section className="metric-rack">
            <div className="primary-metric">
              <span>Sustained throughput</span>
              <strong>{Math.round(slowest)}</strong>
              <b>Mbit/s</b>
              <small>slower direction</small>
            </div>
            <div>
              <span>Median latency</span>
              <strong>
                {result.latencyMedianMs.toFixed(1)} <b>ms</b>
              </strong>
            </div>
            <div>
              <span>p95 latency</span>
              <strong>
                {result.latencyP95Ms.toFixed(1)} <b>ms</b>
              </strong>
            </div>
            {result.jitterMs >= 0 && (
              <div>
                <span>Jitter</span>
                <strong>
                  {result.jitterMs.toFixed(1)} <b>ms</b>
                </strong>
              </div>
            )}
            {result.packetLossPercent >= 0 && (
              <div>
                <span>Packet loss</span>
                <strong>
                  {result.packetLossPercent.toFixed(1)} <b>%</b>
                </strong>
              </div>
            )}
          </section>
          <div className="throughput-detail">
            <span>
              Measured peer download <b>{Math.round(result.downMbps)} Mbit/s</b>
            </span>
            <Meter value={result.downMbps} max={Math.max(result.downMbps, 1000)} />
            <small>One measured direction; not separate up/down links.</small>
          </div>
          {result.windowsProfile && (
            <p className="network-profile-note">
              Windows network profile: <b>{result.windowsProfile}</b> — informational only, it does
              not affect operation.
            </p>
          )}
        </div>
      )}
      <section className="guidance-grid">
        <article>
          <span className="guidance-icon">⌁</span>
          <div>
            <h3>Prefer wired Ethernet</h3>
            <p>
              Connect both nodes to the same switch. A direct 2.5 GbE link can improve larger model
              splits.
            </p>
          </div>
        </article>
        <article>
          <span className="guidance-icon">⌁</span>
          <div>
            <h3>If you use Wi-Fi</h3>
            <p>
              Use 5 GHz or 6 GHz near the access point, pause downloads, and retest after changing
              rooms.
            </p>
          </div>
        </article>
        <article>
          <span className="guidance-icon">⌁</span>
          <div>
            <h3>Direct Ethernet cable</h3>
            <p>
              A cable directly between the two computers works with static 10.10.10.x addresses or
              automatic 169.254.x.x link-local addresses — no router and no network-profile change.
            </p>
          </div>
        </article>
      </section>
    </div>
  );
}
