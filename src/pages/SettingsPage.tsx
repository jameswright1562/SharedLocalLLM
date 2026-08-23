import { useState } from "react";
import type { ReactNode } from "react";
import {
  Badge,
  Box,
  Button,
  Flex,
  Group,
  Paper,
  Stack,
  Switch,
  Tabs,
  Text,
  TextInput,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { IconFolder } from "@tabler/icons-react";

import { StatusBanner } from "../components/StatusBanner";
import { describeAppError } from "../services/errors";
import type { PageProps } from "../types";

export function SettingsPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [tab, setTab] = useState<"general" | "runtime" | "sources" | "logs">("general");
  const [deviceName, setDeviceName] = useState(snapshot.deviceName);
  const [apiPort, setApiPort] = useState(snapshot.apiPort);
  const [authRequired, setAuthRequired] = useState(snapshot.authRequired);
  const [autostart, setAutostart] = useState(snapshot.autostart);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [runtimeProgress, setRuntimeProgress] = useState("");

  async function saveSettings() {
    setBusy(true);
    setMessage("");
    try {
      await service.updateSettings({
        deviceName: deviceName.trim(),
        apiPort,
        authRequired,
        autostart,
      });
      await refreshSnapshot();
      setMessage("Settings saved.");
    } catch (reason) {
      setMessage(describeAppError(reason, "Settings could not be saved."));
    } finally {
      setBusy(false);
    }
  }

  async function addFolder() {
    setBusy(true);
    setMessage("");
    try {
      const directory = await service.addModelDirectory();
      if (!directory) {
        setMessage("No folder selected.");
        return;
      }
      await refreshSnapshot();
      setMessage("Model folder added.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The folder could not be added."));
    } finally {
      setBusy(false);
    }
  }

  async function removeFolder(id: string) {
    setBusy(true);
    setMessage("");
    try {
      await service.removeModelDirectory(id);
      await refreshSnapshot();
      setMessage("Model folder removed.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The folder could not be removed."));
    } finally {
      setBusy(false);
    }
  }

  async function installRuntime() {
    setBusy(true);
    setMessage("");
    try {
      await service.installRuntime((_percent, status) => setRuntimeProgress(status));
      await refreshSnapshot();
      setMessage("Runtime installed.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The runtime could not be installed."));
    } finally {
      setBusy(false);
    }
  }

  async function openLogs() {
    setMessage("");
    try {
      await service.openLogsFolder();
    } catch (reason) {
      setMessage(describeAppError(reason, "The logs folder could not be opened."));
    }
  }

  return (
    <Box>
      <Box mb="lg">
        <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
          Application
        </Text>
        <Title order={1}>Settings &amp; logs</Title>
        <Text c="dimmed">Runtime, local model sources, and diagnostics for this computer.</Text>
      </Box>

      <Tabs value={tab} onChange={(value) => value && setTab(value as typeof tab)}>
        <Tabs.List aria-label="Settings sections" mb="lg">
          <Tabs.Tab value="general">General</Tabs.Tab>
          <Tabs.Tab value="runtime">Runtime</Tabs.Tab>
          <Tabs.Tab value="sources">Model sources</Tabs.Tab>
          <Tabs.Tab value="logs">Logs</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="general">
          <Stack gap="lg">
            <Title order={3}>General</Title>
            <SettingsRow title="Device name" description="Shown to the paired computer.">
              <TextInput
                aria-label="Device name"
                maxLength={80}
                w={280}
                value={deviceName}
                onChange={(event) => setDeviceName(event.target.value)}
              />
            </SettingsRow>
            <SettingsRow
              title="Local API port"
              description="Loopback only. SharedLocalLLM will not silently choose a new port."
            >
              <TextInput
                aria-label="Local API port"
                type="number"
                min={1024}
                max={65535}
                w={130}
                value={apiPort}
                onChange={(event) => setApiPort(Number(event.target.value))}
              />
            </SettingsRow>
            <SettingsRow
              title="Require API key"
              description="When off, local tools can call the loopback API without the bearer key."
            >
              <Switch
                role="switch"
                aria-label="Require API key"
                aria-checked={authRequired}
                checked={authRequired}
                onChange={(event) => setAuthRequired(event.currentTarget.checked)}
              />
            </SettingsRow>
            <SettingsRow title="Start with Windows" description="Launch to the notification area.">
              <Switch
                role="switch"
                aria-label="Start with Windows"
                aria-checked={autostart}
                checked={autostart}
                onChange={(event) => setAutostart(event.currentTarget.checked)}
              />
            </SettingsRow>
            <Group>
              <Button
                disabled={
                  busy ||
                  !deviceName.trim() ||
                  deviceName.trim().length > 80 ||
                  apiPort < 1024 ||
                  apiPort > 65535
                }
                onClick={() => void saveSettings()}
              >
                {busy ? "Saving…" : "Save changes"}
              </Button>
            </Group>
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="runtime">
          <Stack gap="lg">
            <Title order={3}>Runtime</Title>
            <Text c="dimmed" maw={620}>
              Install or repair the pinned llama.cpp CUDA runtime. Failed downloads are never
              activated.
            </Text>
            <SettingsRow
              title="llama.cpp runtime"
              description={
                snapshot.runtime.error ||
                runtimeProgress ||
                "Verified backend used by this computer."
              }
            >
              <Badge color={snapshot.runtime.status === "ready" ? "mint" : "gray"} variant="light">
                {snapshot.runtime.version ?? snapshot.runtime.status}
              </Badge>
            </SettingsRow>
            <Group>
              <Button disabled={busy} onClick={() => void installRuntime()}>
                {busy
                  ? "Installing…"
                  : snapshot.runtime.status === "ready"
                    ? "Repair runtime"
                    : "Install runtime"}
              </Button>
            </Group>
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="sources">
          <Stack gap="md">
            <Flex justify="space-between" align="flex-end" gap="md" wrap="wrap">
              <div>
                <Title order={3}>Model sources</Title>
                <Text size="sm" c="dimmed">
                  Directories are read-only and configured per computer.
                </Text>
              </div>
              <Button disabled={busy} onClick={() => void addFolder()}>
                Add folder
              </Button>
            </Flex>
            <Stack gap="xs">
              {snapshot.modelDirectories.map((directory) => (
                <Paper key={directory.id} p="sm" bg="dark.8" withBorder>
                  <Flex align="center" gap="sm" wrap="wrap">
                    <ThemeIcon variant="light" color="cyan">
                      {directory.source === "lm-studio" ? (
                        <Text size="10px" ff="monospace" fw={700}>
                          LM
                        </Text>
                      ) : (
                        <IconFolder size={15} />
                      )}
                    </ThemeIcon>
                    <Box style={{ flex: 1, minWidth: 200 }}>
                      <Text size="sm" fw={600}>
                        {directory.path}
                      </Text>
                      <Text size="xs" c="dimmed">
                        {directory.source === "lm-studio"
                          ? "LM Studio · automatic"
                          : "Custom folder"}
                      </Text>
                    </Box>
                    {directory.source === "custom" && (
                      <Button
                        variant="subtle"
                        color="coral"
                        size="compact-sm"
                        disabled={busy}
                        onClick={() => void removeFolder(directory.id)}
                      >
                        Remove
                      </Button>
                    )}
                  </Flex>
                </Paper>
              ))}
            </Stack>
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="logs">
          <Stack gap="md">
            <Flex justify="space-between" align="flex-end" gap="md" wrap="wrap">
              <div>
                <Title order={3}>Live logs</Title>
                <Text size="sm" c="dimmed">
                  Secrets, prompt text, and personal path prefixes are redacted.
                </Text>
              </div>
              <Button variant="default" onClick={() => void openLogs()}>
                Open logs folder
              </Button>
            </Flex>
            <Paper
              component="pre"
              className="log-viewer"
              aria-label="Application logs"
              withBorder
              p="md"
              fz="xs"
              style={{
                fontFamily: "var(--mantine-font-family-monospace)",
                margin: 0,
                whiteSpace: "pre-wrap",
              }}
            >
              {snapshot.logs.length ? snapshot.logs.join("\n") : "No log entries in this session."}
            </Paper>
          </Stack>
        </Tabs.Panel>
      </Tabs>

      {message && <StatusBanner message={message} />}
    </Box>
  );
}

function SettingsRow({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <Flex justify="space-between" align="center" gap="xl" wrap="wrap">
      <Box maw={420}>
        <Text fw={600}>{title}</Text>
        <Text size="sm" c="dimmed">
          {description}
        </Text>
      </Box>
      {children}
    </Flex>
  );
}
