import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Code,
  Flex,
  Group,
  Loader,
  Paper,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { IconEye, IconEyeOff, IconPlugOff, IconServer } from "@tabler/icons-react";

import { describeAppError } from "../services/errors";
import type { ApiConfig, ApiTryResult, PageProps } from "../types";

function maskApiKey(key: string) {
  const prefix = key.startsWith("sk-local-") ? "sk-local-" : key.slice(0, 4);
  return `${prefix}••••••••••`;
}

function formatBody(body: string) {
  try {
    return JSON.stringify(JSON.parse(body), null, 2);
  } catch {
    return body;
  }
}

export function ApiPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [config, setConfig] = useState<ApiConfig | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState<"url" | "key" | "curl" | "">("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [trying, setTrying] = useState(false);
  const [tryResult, setTryResult] = useState<ApiTryResult | null>(null);
  useEffect(() => {
    let active = true;
    void service
      .getApiConfig()
      .then((nextConfig) => {
        if (active) setConfig(nextConfig);
      })
      .catch((reason) => {
        if (active) setError(describeAppError(reason, "Could not read the API configuration."));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [service]);
  async function copy(value: string, what: "url" | "key" | "curl") {
    setError("");
    try {
      if (!navigator.clipboard) throw new Error("Clipboard is unavailable in this environment.");
      await navigator.clipboard.writeText(value);
      setCopied(what);
      window.setTimeout(() => setCopied(""), 1400);
    } catch (reason) {
      setError(describeAppError(reason, "The value could not be copied."));
    }
  }
  async function regenerate() {
    if (snapshot.cluster.status === "running") {
      const confirmed = window.confirm(
        "Regenerating the API key stops the running cluster. Continue?",
      );
      if (!confirmed) return;
    }
    setBusy(true);
    setError("");
    try {
      setConfig(await service.regenerateApiKey());
      setRevealed(true);
      await refreshSnapshot();
    } catch (reason) {
      setError(describeAppError(reason, "The API key could not be regenerated."));
    } finally {
      setBusy(false);
    }
  }
  async function runExample() {
    setTrying(true);
    setTryResult(null);
    setError("");
    try {
      setTryResult(await service.tryApiRequest());
    } catch (reason) {
      setError(describeAppError(reason, "The example request could not be completed."));
    } finally {
      setTrying(false);
    }
  }
  const curlPayload = JSON.stringify({
    model: "local-model",
    messages: [{ role: "user", content: "Hello" }],
  });
  const authHeader = config?.authRequired ? ` -H "Authorization: Bearer ${config.apiKey}"` : "";
  const curl = config
    ? `curl.exe -X POST "${config.url}/v1/chat/completions"${authHeader} -H "Content-Type: application/json" --data-raw '${curlPayload}'`
    : "";
  return (
    <Box>
      <Box mb="lg">
        <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
          Loopback interface
        </Text>
        <Title order={1}>Local API</Title>
        <Text c="dimmed">
          Connect local tools using an OpenAI-compatible endpoint. It is never exposed to the LAN.
        </Text>
      </Box>

      {error && (
        <Alert role="alert" variant="light" color="coral" mb="md">
          {error}
        </Alert>
      )}

      {loading ? (
        <Group gap="xs" role="status" aria-live="polite">
          <Loader size="xs" type="dots" />
          <Text>Reading API configuration…</Text>
        </Group>
      ) : !config ? (
        <Paper withBorder p="xl">
          <Stack align="center" gap="xs" ta="center">
            <ThemeIcon variant="light" color="amber" size="xl" radius="xl">
              !
            </ThemeIcon>
            <Title order={3}>API configuration unavailable</Title>
            <Text c="dimmed">Resolve the error above, then reopen this page.</Text>
          </Stack>
        </Paper>
      ) : (
        <>
          <Paper withBorder p="lg" mb="md">
            <Flex justify="space-between" align="center" gap="md" wrap="wrap">
              <Group gap="md" wrap="nowrap">
                <ThemeIcon
                  variant="light"
                  size="lg"
                  radius="xl"
                  color={config.healthy ? "mint" : "coral"}
                >
                  {config.healthy ? <IconServer size={22} /> : <IconPlugOff size={22} />}
                </ThemeIcon>
                <div>
                  <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
                    Connection health
                  </Text>
                  <Title order={3}>{config.healthy ? "Listening on loopback" : "API unavailable"}</Title>
                  <Text size="sm" c="dimmed">
                    {config.healthy
                      ? "Requests from this computer can reach the active coordinator."
                      : "Start the API service or resolve the port conflict in Settings."}
                  </Text>
                </div>
              </Group>
              <Badge color={config.healthy ? "mint" : "gray"} variant="light">
                {config.healthy ? "Healthy" : "Offline"}
              </Badge>
            </Flex>
          </Paper>

          <Paper withBorder p="lg" mb="md">
            <Stack gap="sm">
              <CredentialRow label="Base URL">
                <Code style={{ flex: 1 }}>{config.url}</Code>
                <Button variant="default" size="compact-sm" onClick={() => void copy(config.url, "url")}>
                  {copied === "url" ? "Copied" : "Copy"}
                </Button>
              </CredentialRow>
              <CredentialRow label="API key">
                <Code style={{ flex: 1 }}>
                  {revealed ? config.apiKey : maskApiKey(config.apiKey)}
                </Code>
                <ActionIcon
                  variant="default"
                  size="md"
                  aria-label={revealed ? "Hide API key" : "Show API key"}
                  onClick={() => setRevealed(!revealed)}
                >
                  {revealed ? <IconEyeOff size={15} /> : <IconEye size={15} />}
                </ActionIcon>
                <Button variant="default" size="compact-sm" onClick={() => void copy(config.apiKey, "key")}>
                  {copied === "key" ? "Copied" : "Copy"}
                </Button>
              </CredentialRow>
              <CredentialRow label="Authentication">
                <Code style={{ flex: 1 }}>
                  {config.authRequired ? "Bearer key required" : "Disabled — open access"}
                </Code>
              </CredentialRow>
              <Flex justify="space-between" align="center" gap="md" wrap="wrap" mt="xs">
                <Text size="xs" c="dimmed" maw={520}>
                  {config.authRequired
                    ? "Keep this key private. Regenerating it immediately invalidates the previous key."
                    : "Key checks are off in Settings, so any local tool can call this API without a key."}
                </Text>
                <Button
                  variant="subtle"
                  color="coral"
                  size="compact-sm"
                  disabled={busy}
                  onClick={() => void regenerate()}
                >
                  {busy ? "Regenerating…" : "Regenerate key"}
                </Button>
              </Flex>
            </Stack>
          </Paper>

          <Paper withBorder mb="md" style={{ overflow: "hidden" }}>
            <Group justify="space-between" p="md" bg="dark.8" wrap="nowrap" gap="md">
              <div>
                <Text size="10px" ff="monospace" tt="uppercase" lts={2} c="cyan" fw={600}>
                  PowerShell / curl
                </Text>
                <Title order={4}>Chat completion</Title>
              </div>
              <Group gap="sm" wrap="nowrap">
                <Button size="compact-sm" disabled={trying} onClick={() => void runExample()}>
                  {trying ? "Running…" : "Try it"}
                </Button>
                <Button variant="default" size="compact-sm" onClick={() => void copy(curl, "curl")}>
                  {copied === "curl" ? "Copied" : "Copy example"}
                </Button>
              </Group>
            </Group>
            <Box
              component="pre"
              m={0}
              p="md"
              fz="sm"
              lh={1.6}
              style={{
                fontFamily: "var(--mantine-font-family-monospace)",
                overflowX: "auto",
                whiteSpace: "pre-wrap",
              }}
            >
              <code>{curl}</code>
            </Box>
            {tryResult && (
              <Box aria-label="Example response" p="md" pt={0}>
                <Group gap="sm" mb="xs">
                  <Badge
                    color={tryResult.status >= 200 && tryResult.status < 400 ? "mint" : "coral"}
                    variant="light"
                  >
                    HTTP {tryResult.status}
                  </Badge>
                  <Text size="xs" c="dimmed">
                    {tryResult.durationMs} ms
                  </Text>
                </Group>
                <Box
                  component="pre"
                  m={0}
                  p="sm"
                  fz="xs"
                  bg="dark.9"
                  style={{
                    borderRadius: "var(--mantine-radius-xs)",
                    border: "1px solid var(--mantine-color-dark-5)",
                    fontFamily: "var(--mantine-font-family-monospace)",
                    overflowX: "auto",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  <code>{formatBody(tryResult.body)}</code>
                </Box>
              </Box>
            )}
          </Paper>
        </>
      )}

      <Paper withBorder p="lg">
        <Title order={3} mb="md">
          Supported endpoints
        </Title>
        <Stack gap="xs">
          <EndpointRow method="GET" path="/health" note="Service and model readiness" />
          <EndpointRow method="GET" path="/v1/models" note="Available local models" />
          <EndpointRow method="POST" path="/v1/chat/completions" note="Chat, images, and streaming" />
          <EndpointRow method="POST" path="/v1/completions" note="Text completions and streaming" />
        </Stack>
      </Paper>
    </Box>
  );
}

function CredentialRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Flex align="center" gap="sm" wrap="wrap">
      <Text size="xs" c="dimmed" tt="uppercase" lts={1} fw={600} w={120}>
        {label}
      </Text>
      {children}
    </Flex>
  );
}

function EndpointRow({ method, path, note }: { method: string; path: string; note: string }) {
  return (
    <Flex align="center" gap="sm" wrap="wrap">
      <Badge variant="outline" color="cyan" ff="monospace" w={52} ta="center">
        {method}
      </Badge>
      <Code>{path}</Code>
      <Text size="sm" c="dimmed">
        {note}
      </Text>
    </Flex>
  );
}
