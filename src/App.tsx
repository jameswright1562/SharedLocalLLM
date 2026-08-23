import { useCallback, useEffect, useState } from "react";
import type { ReactElement } from "react";
import {
  Alert,
  AppShell,
  Badge,
  Box,
  Burger,
  Button,
  Group,
  Loader,
  NavLink,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import {
  IconApi,
  IconArrowsExchange,
  IconBox,
  IconGauge,
  IconHome,
  IconMessage,
  IconServer2,
  IconSettings,
} from "@tabler/icons-react";

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

const navigation: Array<{
  id: PageId;
  label: string;
  icon: typeof IconHome;
}> = [
  { id: "overview", label: "Overview", icon: IconHome },
  { id: "nodes", label: "Nodes", icon: IconServer2 },
  { id: "network", label: "Network", icon: IconArrowsExchange },
  { id: "models", label: "Models", icon: IconBox },
  { id: "benchmarks", label: "Benchmarks", icon: IconGauge },
  { id: "chat", label: "Chat", icon: IconMessage },
  { id: "api", label: "API", icon: IconApi },
  { id: "settings", label: "Settings", icon: IconSettings },
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

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span />
      <span />
    </div>
  );
}

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
      <Stack align="center" justify="center" mih="100vh" gap="lg">
        <BrandMark />
        <Title order={1}>SharedLocalLLM</Title>
        {loadingError ? (
          <Stack align="center" gap="md" w={420} maw="90vw">
            <Alert color="coral" variant="light" role="alert" w="100%">
              {loadingError}
            </Alert>
            <Button onClick={() => void refreshSnapshot()}>Try again</Button>
          </Stack>
        ) : (
          <Group gap="xs">
            <Loader size="xs" type="dots" />
            <Text c="dimmed">Reading local capabilities…</Text>
          </Group>
        )}
      </Stack>
    );
  }

  if (!snapshot.setupComplete) {
    return (
      <>
        {service === demoService && (
          <Alert variant="light" color="cyan" role="status" radius={0}>
            Browser preview — simulated hardware. This is not a live two-computer cluster.
          </Alert>
        )}
        <SetupWizard snapshot={snapshot} service={service} onComplete={setSnapshot} />
      </>
    );
  }

  const CurrentPage = pageComponents[page];
  const onlineNodes = snapshot.nodes.filter((node) => node.online).length;

  const preview = service === demoService;

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 248, breakpoint: "sm", collapsed: { mobile: !sidebarOpen } }}
      padding="md"
    >
      <AppShell.Header px="md">
        <Group h="100%" justify="space-between" wrap="nowrap" gap="md">
          <Group gap="sm" wrap="nowrap">
            <Burger
              opened={sidebarOpen}
              onClick={() => setSidebarOpen(!sidebarOpen)}
              hiddenFrom="sm"
              size="sm"
              aria-label="Toggle navigation"
            />
            <ComputePath cluster={snapshot.cluster} nodes={snapshot.nodes} compact />
          </Group>
          <Group gap="sm" wrap="nowrap" visibleFrom="xs">
            <Text size="sm" c="dimmed" style={{ whiteSpace: "nowrap" }}>
              {onlineNodes}/2 nodes online
            </Text>
            <Text size="sm" tt="capitalize" c="dimmed" visibleFrom="md" style={{ whiteSpace: "nowrap" }}>
              {snapshot.network?.classification ?? "link untested"}
            </Text>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar component="nav" aria-label="Primary navigation" p={0}>
        <AppShell.Section p="md" pb="sm">
          <Group gap="sm" wrap="nowrap">
            <BrandMark />
            <Box>
              <Text fw={700} lh={1.1}>
                SharedLocal
              </Text>
              <Text size="xs" c="cyan" ff="monospace" fw={600} lh={1.4}>
                LLM · LOCAL ONLY
              </Text>
            </Box>
          </Group>
        </AppShell.Section>
        <AppShell.Section grow component="div" px="xs">
          {navigation.map((item) => (
            <NavLink
              key={item.id}
              component="button"
              type="button"
              active={page === item.id}
              label={<Text size="sm">{item.label}</Text>}
              leftSection={<item.icon size={17} stroke={1.6} />}
              rightSection={
                item.id === "nodes" ? (
                  <Badge size="xs" variant="light" color="cyan" aria-hidden>
                    {onlineNodes}
                  </Badge>
                ) : undefined
              }
              onClick={() => navigate(item.id)}
            />
          ))}
        </AppShell.Section>
        <AppShell.Section p="md" pt="sm">
          <Group gap="xs" wrap="nowrap" mb="xs">
            <span
              aria-hidden="true"
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                flex: "0 0 auto",
                background:
                  snapshot.runtime.status === "ready"
                    ? "var(--mantine-color-mint-4)"
                    : "var(--mantine-color-amber-4)",
              }}
            />
            <Box>
              <Text size="xs" fw={600}>
                Runtime {snapshot.runtime.status}
              </Text>
              <Text size="10px" c="dimmed">
                {snapshot.runtime.version ?? "Action required"}
              </Text>
            </Box>
          </Group>
          <Text size="9px" c="dimmed" tt="uppercase" lts={2}>
            SHAREDLOCALLLM · LOCAL ONLY
          </Text>
        </AppShell.Section>
      </AppShell.Navbar>

      <AppShell.Main>
        {preview && (
          <Alert variant="light" color="amber" role="status" mb="md">
            Browser preview — simulated hardware. This is not a live two-computer cluster.
          </Alert>
        )}
        {loadingError && (
          <Alert variant="light" color="coral" role="alert" mb="md">
            Latest refresh failed: {loadingError}
          </Alert>
        )}
        <CurrentPage
          snapshot={snapshot}
          service={service}
          refreshSnapshot={refreshSnapshot}
          navigate={navigate}
        />
      </AppShell.Main>
    </AppShell>
  );
}
