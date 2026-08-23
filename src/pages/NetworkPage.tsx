import { useState } from "react";
import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Flex,
  Group,
  Paper,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { IconActivity, IconBolt } from "@tabler/icons-react";

import { describeAppError } from "../services/errors";
import type { NetworkBenchmark, PageProps } from "../types";

const verdictCopy: Record<NetworkBenchmark["classification"], string> = {
  good: "This path is well suited to layer-split inference.",
  usable: "Distributed inference should work, but compare it with single-node placement.",
  poor: "Prefer single-node inference when the model fits. The link may constrain token speed.",
};

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
    <Box>
      <Flex justify="space-between" align="flex-start" gap="md" wrap="wrap" mb="lg">
        <Box>
          <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
            Transport
          </Text>
          <Title order={1}>Link diagnostics</Title>
          <Text c="dimmed">Measure the peer route used for distributed inference.</Text>
        </Box>
        <Button disabled={testing || snapshot.nodes.length < 2} onClick={() => void runTest()}>
          {testing ? "Testing the peer channel…" : "Run network test"}
        </Button>
      </Flex>

      {error && (
        <Alert role="alert" variant="light" color="coral" title="Test failed" mb="md">
          {error}
        </Alert>
      )}

      {!result ? (
        <Paper withBorder p="xl">
          <Stack align="center" gap="xs" ta="center">
            <Text size="28px" c="cyan">
              ↔
            </Text>
            <Title order={3}>No link result yet</Title>
            <Text c="dimmed" maw={420}>
              Pair a worker, close large transfers, and run the test to get a topology
              recommendation.
            </Text>
          </Stack>
        </Paper>
      ) : (
        <div data-testid="network-test-result">
          <Paper
            withBorder
            p="xl"
            bg="dark.8"
            mb="md"
            component="section"
            aria-label={`Link classification ${result.classification}`}
          >
            <Group gap="md" mb="xs" wrap="nowrap">
              <ThemeIcon variant="light" size="xl" color={verdictColor(result.classification)}>
                <IconActivity size={22} />
              </ThemeIcon>
              <div>
                <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
                  Link classification
                </Text>
                <Text className="classification" tt="capitalize" fw={700} size="xl" lh={1.2}>
                  {result.classification}
                </Text>
              </div>
            </Group>
            <Text c="dimmed">{verdictCopy[result.classification]}</Text>
          </Paper>

          <SimpleGrid
            cols={{ base: 2, sm: 3 }}
            spacing="sm"
            mb="md"
            component="section"
            aria-label="Measured link metrics"
          >
            <MetricCard
              label="Sustained throughput"
              value={`${Math.round(slowest)}`}
              unit="Mbit/s"
              note="slower direction"
              primary
            />
            <MetricCard
              label="Median latency"
              value={result.latencyMedianMs.toFixed(1)}
              unit="ms"
            />
            <MetricCard label="p95 latency" value={result.latencyP95Ms.toFixed(1)} unit="ms" />
            {result.jitterMs >= 0 && (
              <MetricCard label="Jitter" value={result.jitterMs.toFixed(1)} unit="ms" />
            )}
            {result.packetLossPercent >= 0 && (
              <MetricCard
                label="Packet loss"
                value={result.packetLossPercent.toFixed(1)}
                unit="%"
              />
            )}
          </SimpleGrid>

          <Paper withBorder p="md" mb="md">
            <Text size="sm" mb={4}>
              Measured peer download <b>{Math.round(result.downMbps)} Mbit/s</b>
            </Text>
            <Progress
              value={(result.downMbps / Math.max(result.downMbps, 1000)) * 100}
              color="cyan"
              size="sm"
              radius="xs"
              mb={4}
            />
            <Text size="xs" c="dimmed">
              One measured direction; not separate up/down links.
            </Text>
          </Paper>

          {result.windowsProfile && (
            <Text size="sm" c="dimmed">
              Windows network profile: <b>{result.windowsProfile}</b> — informational only, it does
              not affect operation.
            </Text>
          )}
        </div>
      )}

      <SimpleGrid
        cols={{ base: 1, sm: 3 }}
        spacing="md"
        mt="xl"
        component="section"
        aria-label="Link guidance"
      >
        <GuidanceCard title="Prefer wired Ethernet">
          Connect both nodes to the same switch. A direct 2.5 GbE link can improve larger model
          splits.
        </GuidanceCard>
        <GuidanceCard title="If you use Wi-Fi">
          Use 5 GHz or 6 GHz near the access point, pause downloads, and retest after changing
          rooms.
        </GuidanceCard>
        <GuidanceCard title="Direct Ethernet cable">
          A cable directly between the two computers works with static 10.10.10.x addresses or
          automatic 169.254.x.x link-local addresses — no router and no network-profile change.
        </GuidanceCard>
      </SimpleGrid>
    </Box>
  );
}

function verdictColor(classification: NetworkBenchmark["classification"]) {
  if (classification === "good") return "mint";
  if (classification === "usable") return "amber";
  return "coral";
}

function MetricCard({
  label,
  value,
  unit,
  note,
  primary,
}: {
  label: string;
  value: string;
  unit: string;
  note?: string;
  primary?: boolean;
}) {
  return (
    <Paper withBorder p="md">
      <Text size="xs" tt="uppercase" lts={1} c="dimmed" fw={600}>
        {label}
      </Text>
      <Group gap={4} align="baseline">
        <Text size={primary ? "30px" : "lg"} fw={700}>
          {value}
        </Text>
        <Badge variant="light" size="sm" color="cyan">
          {unit}
        </Badge>
      </Group>
      {note && (
        <Text size="10px" c="dimmed">
          {note}
        </Text>
      )}
    </Paper>
  );
}

function GuidanceCard({ title, children }: { title: string; children: string }) {
  return (
    <Card withBorder p="md" component="article">
      <Stack gap="xs">
        <Group gap="sm">
          <ThemeIcon variant="light" color="amber" size="md">
            <IconBolt size={14} />
          </ThemeIcon>
          <Title order={4}>{title}</Title>
        </Group>
        <Text size="sm" c="dimmed">
          {children}
        </Text>
      </Stack>
    </Card>
  );
}
