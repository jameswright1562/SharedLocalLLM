import { useCallback, useEffect, useState } from "react";
import type { ReactElement } from "react";

import { ComputePath } from "./components/ComputePath";
import { SetupWizard } from "./components/SetupWizard";
import { ApiPage } from "./pages/ApiPage";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { ChatPage } from "./pages/ChatPage";
import { ModelsPage } from "./pages/ModelsPage";
import { NetworkPage } from "./pages/NetworkPage";
import { NodesPage } from "./pages/NodesPage";
import { OverviewPage } from "./pages/OverviewPage";
import { SettingsPage } from "./pages/SettingsPage";
import { appService, demoService } from "./services/appService";
import { describeAppError } from "./services/errors";
import type { AppService, AppSnapshot, PageId, PageProps } from "./types";

const navigation: Array<{ id: PageId; label: string; icon: string }> = [
  { id: "overview", label: "Overview", icon: "⌂" },
  { id: "nodes", label: "Nodes", icon: "▦" },
  { id: "network", label: "Network", icon: "↔" },
  { id: "models", label: "Models", icon: "◫" },
  { id: "benchmarks", label: "Benchmarks", icon: "⌁" },
  { id: "chat", label: "Chat", icon: "◌" },
  { id: "api", label: "API", icon: "{ }" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

const pageComponents: Record<PageId, (props: PageProps) => ReactElement> = {
  overview: OverviewPage,
  nodes: NodesPage,
  network: NetworkPage,
  models: ModelsPage,
  benchmarks: BenchmarksPage,
  chat: ChatPage,
  api: ApiPage,
  settings: SettingsPage,
};

export default function App({ service = appService }: { service?: AppService }) {
  const [snapshot, setSnapshot] = useState<AppSnapshot | null>(null);
  const [page, setPage] = useState<PageId>("overview");
  const [loadingError, setLoadingError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refreshSnapshot = useCallback(async () => {
    try {
      setSnapshot(await service.getAppSnapshot());
      setLoadingError("");
    } catch (reason) {
      setLoadingError(
        describeAppError(reason, "SharedLocalLLM could not read the local service state."),
      );
    }
  }, [service]);

  useEffect(() => {
    // Initial hydration is the boundary between the native service and the UI.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshSnapshot();
  }, [refreshSnapshot]);

  useEffect(() => {
    let active = true;
    let timer = 0;
    const schedule = () => {
      timer = window.setTimeout(() => {
        void refreshSnapshot().finally(() => {
          if (active) schedule();
        });
      }, 8_000);
    };
    schedule();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [refreshSnapshot]);

  function navigate(nextPage: PageId) {
    setPage(nextPage);
    setSidebarOpen(false);
  }

  if (!snapshot) {
    return (
      <main className="boot-screen">
        <div className="boot-mark">
          <span />
          <span />
        </div>
        <h1>SharedLocalLLM</h1>
        {loadingError ? (
          <>
            <p role="alert">{loadingError}</p>
            <button className="button primary" onClick={() => void refreshSnapshot()}>
              Try again
            </button>
          </>
        ) : (
          <p>
            <span className="spinner" /> Reading local capabilities…
          </p>
        )}
      </main>
    );
  }

  if (!snapshot.setupComplete) {
    return (
      <>
        {service === demoService && (
          <div className="preview-banner" role="status">
            Browser preview — simulated hardware. This is not a live two-computer cluster.
          </div>
        )}
        <SetupWizard snapshot={snapshot} service={service} onComplete={setSnapshot} />
      </>
    );
  }

  const CurrentPage = pageComponents[page];
  const onlineNodes = snapshot.nodes.filter((node) => node.online).length;

  const preview = service === demoService;

  return (
    <div className="app-shell">
      {preview && (
        <div className="preview-banner" role="status">
          Browser preview — simulated hardware. This is not a live two-computer cluster.
        </div>
      )}
      {loadingError && snapshot && (
        <div className="preview-banner" role="alert">
          Latest refresh failed: {loadingError}
        </div>
      )}
      <button
        className="mobile-menu"
        aria-label="Open navigation"
        aria-expanded={sidebarOpen}
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        ☰
      </button>
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <header className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
          </div>
          <div>
            <strong>SharedLocal</strong>
            <span>LLM</span>
          </div>
        </header>
        <nav aria-label="Primary navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "active" : ""}
              aria-current={page === item.id ? "page" : undefined}
              onClick={() => navigate(item.id)}
            >
              <span className="nav-icon" aria-hidden="true">
                {item.icon}
              </span>
              <span>{item.label}</span>
              {item.id === "nodes" && <i>{onlineNodes}</i>}
            </button>
          ))}
        </nav>
        <footer>
          <div className="sidebar-runtime">
            <span
              className={`status-dot ${snapshot.runtime.status === "ready" ? "ready" : "warning"}`}
              aria-hidden="true"
            />
            <div>
              <strong>Runtime {snapshot.runtime.status}</strong>
              <small>{snapshot.runtime.version ?? "Action required"}</small>
            </div>
          </div>
          <span className="app-version">SHAREDLOCALLLM · LOCAL ONLY</span>
        </footer>
      </aside>
      {sidebarOpen && (
        <button
          className="sidebar-scrim"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <div className="app-main">
        <header className="topbar">
          <ComputePath cluster={snapshot.cluster} nodes={snapshot.nodes} compact />
          <div className="top-status">
            <span>{onlineNodes}/2 nodes online</span>
            <i aria-hidden="true" />
            <span className="capitalize">
              {snapshot.network?.classification ?? "link untested"}
            </span>
          </div>
        </header>
        <main className="content">
          <CurrentPage
            snapshot={snapshot}
            service={service}
            refreshSnapshot={refreshSnapshot}
            navigate={navigate}
          />
        </main>
      </div>
    </div>
  );
}
