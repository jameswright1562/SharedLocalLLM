import {
  Alert,
  Box,
  Button,
  Card,
  Flex,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { IconAlertTriangle, IconPlus } from "@tabler/icons-react";

import type { PageProps } from "../types";
import { ComputePath } from "../components/ComputePath";
import { Meter, StatusPill } from "../components/Telemetry";
import { fitLabels, formatGb } from "./pageFormat";

export function OverviewPage({ snapshot, service, refreshSnapshot, navigate }: PageProps) {
  const online = snapshot.nodes.filter((node) => node.online);
  const combinedVram = online.reduce((sum, node) => sum + node.gpu.vramAvailableGb, 0);
  const combinedRam = online.reduce((sum, node) => sum + node.ramAvailableGb, 0);
  const clusterModel = snapshot.models.find((model) => model.id === snapshot.cluster.modelId);
  const running = snapshot.cluster.status === "running" || snapshot.cluster.status === "loading";

  return (
    <Box>
      <Flex
        justify="space-between"
        align="flex-start"
        gap="md"
        wrap="wrap"
        mb="lg"
      >
        <Box>
          <Text size="xs" fw={700} tt="uppercase" ls={1.5} c="cyan">
            Control plane
          </Text>
          <Title order={1}>Cluster overview</Title>
          <Text c="dimmed">Live capacity and routing across this trusted pair.</Text>
        </Box>
        <Group gap="sm">
          {running && (
            <Button color="coral" variant="light" onClick={() => void service.stopCluster().then(() => refreshSnapshot())}>
              Stop cluster
            </Button>
          )}
          <Button variant="default" onClick={() => navigate("models")}>
            Choose model
          </Button>
        </Group>
      </Flex>

      {snapshot.cluster.error && (
        <Alert
          role="alert"
          variant="light"
          color="coral"
          title="Cluster stopped"
          icon={<IconAlertTriangle size={18} />}
          mb="md"
        >
          {snapshot.cluster.error}
        </Alert>
      )}

      <ComputePath cluster={snapshot.cluster} nodes={snapshot.nodes} />

      <SimpleGrid cols={{ base: 1, xs: 2, lg: 4 }} spacing="sm" mt="lg" as="section" aria-label="Cluster summary">
        <SummaryStat
          label="Usable GPU memory"
          value={formatGb(combinedVram)}
          note={`Across ${online.length} online ${online.length === 1 ? "node" : "nodes"}`}
        />
        <SummaryStat
          label="Available system memory"
          value={formatGb(combinedRam)}
          note="Currently free on online computers"
        />
        <SummaryStat
          label="Active model"
          value={clusterModel?.name ?? "None loaded"}
          muted={!clusterModel}
          note={clusterModel ? fitLabels[clusterModel.fit] : "Choose a model to begin"}
        />
        <SummaryStat
          label="Link quality"
          value={snapshot.network?.classification ?? "Untested"}
          capitalize
          note={
            snapshot.network
              ? `${Math.round(Math.min(snapshot.network.downMbps, snapshot.network.upMbps))} Mbit/s · ${snapshot.network.latencyP95Ms} ms p95`
              : "Run link diagnostics"
          }
        />
      </SimpleGrid>

      <Flex justify="space-between" align="flex-end" gap="md" wrap="wrap" mt="xl" mb="sm">
        <Box>
          <Text size="xs" fw={700} tt="uppercase" ls={1.5} c="cyan">
            Resource map
          </Text>
          <Title order={2}>Compute nodes</Title>
        </Box>
        <Button variant="subtle" size="compact-sm" onClick={() => navigate("nodes")}>
          Inspect hardware →
        </Button>
      </Flex>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        {snapshot.nodes.slice(0, 2).map((node) => (
          <Card key={node.id} withBorder p="lg" component="article">
            <Card.Section withBorder inheritPadding py="sm" mb="md">
              <Group justify="space-between" wrap="nowrap">
                <Box>
                  <Text size="xs" tt="uppercase" ls={1.5} c="cyan" fw={600}>
                    {node.role}
                  </Text>
                  <Title order={3}>{node.name}</Title>
                </Box>
                <StatusPill online={node.online}>{node.online ? "Online" : "Offline"}</StatusPill>
              </Group>
            </Card.Section>
            <Stack gap="xs">
              <Group justify="space-between" align="baseline">
                <Group gap="xs">
                  <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
                    GPU
                  </Text>
                  <Text fw={600}>{node.gpu.name}</Text>
                </Group>
              </Group>
              <ResourceRow label="Free VRAM" free={formatGb(node.gpu.vramAvailableGb)} total={`${formatGb(node.gpu.vramTotalGb)} total`} />
              <Meter value={node.gpu.vramAvailableGb} max={node.gpu.vramTotalGb} />
              <ResourceRow label="Free RAM" free={formatGb(node.ramAvailableGb)} total={`${formatGb(node.ramTotalGb)} total`} />
              <Meter value={node.ramAvailableGb} max={node.ramTotalGb} tone="amber" />
              <Group justify="space-between" mt="xs">
                <Text size="xs" c="dimmed">
                  {node.adapter.name}
                </Text>
                <Text size="xs" c="dimmed">
                  {node.adapter.linkSpeedMbps
                    ? `${node.adapter.linkSpeedMbps} Mbit/s link`
                    : "Link speed unknown"}
                </Text>
              </Group>
            </Stack>
          </Card>
        ))}
        {snapshot.nodes.length < 2 && (
          <Card withBorder p="lg" component="article" style={{ borderStyle: "dashed" }}>
            <Stack align="center" gap="xs" ta="center" py="md">
              <IconPlus size={30} stroke={1.4} aria-hidden color="var(--mantine-color-dimmed)" />
              <Title order={3}>Pair a second computer</Title>
              <Text c="dimmed" maw={380}>
                Add another node to combine GPU memory or move work to the faster machine.
              </Text>
              <Button variant="default" mt="xs" onClick={() => navigate("nodes")}>
                Open pairing
              </Button>
            </Stack>
          </Card>
        )}
      </SimpleGrid>
    </Box>
  );
}

function SummaryStat({
  label,
  value,
  note,
  muted,
  capitalize,
}: {
  label: string;
  value: string;
  note: string;
  muted?: boolean;
  capitalize?: boolean;
}) {
  return (
    <Paper withBorder p="md">
      <Text size="xs" tt="uppercase" ls={1} c="dimmed" fw={600}>
        {label}
      </Text>
      <Text
        size="lg"
        fw={700}
        tt={capitalize ? "capitalize" : undefined}
        c={muted ? "dimmed" : undefined}
        truncate
      >
        {value}
      </Text>
      <Text size="xs" c="dimmed">
        {note}
      </Text>
    </Paper>
  );
}

function ResourceRow({ label, free, total }: { label: string; free: string; total: string }) {
  return (
    <Group justify="space-between" align="baseline" wrap="nowrap">
      <Group gap="xs">
        <Text size="xs" c="dimmed">
          {label}
        </Text>
        <Text fw={600}>{free}</Text>
      </Group>
      <Text size="xs" c="dimmed">
        {total}
      </Text>
    </Group>
  );
}
