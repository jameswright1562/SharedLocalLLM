import { Badge, Box, Group, Text } from "@mantine/core";
import type { ClusterSession, NodeCapabilities } from "../types";

interface ComputePathProps {
  cluster: ClusterSession;
  nodes: NodeCapabilities[];
  compact?: boolean;
}

const statusColors: Record<ClusterSession["status"], string> = {
  idle: "dark",
  pairing: "amber",
  loading: "cyan",
  ready: "cyan",
  running: "mint",
  stopping: "amber",
  error: "coral",
};

function nodeName(nodes: NodeCapabilities[], id?: string, fallback = "Waiting for peer") {
  return nodes.find((node) => node.id === id)?.name ?? fallback;
}

function PathLine({ active }: { active: boolean }) {
  return (
    <Box
      aria-hidden
      w={28}
      h={1}
      style={{
        flex: "0 0 auto",
        background: active
          ? "linear-gradient(90deg, var(--mantine-color-cyan-6), var(--mantine-color-cyan-4))"
          : "var(--mantine-color-dark-4)",
      }}
    />
  );
}

export function ComputePath({ cluster, nodes, compact = false }: ComputePathProps) {
  const active = cluster.status === "loading" || cluster.status === "running";
  const coordinator = nodeName(nodes, cluster.coordinatorNodeId, "Choose coordinator");
  const worker = nodeName(nodes, cluster.workerNodeId);

  const renderNode = (index: string, label: string, value: string, empty = false) => (
    <Box maw={compact ? 150 : undefined}>
      <Text size="10px" fw={600} c="dimmed" lh={1}>
        {index}
      </Text>
      <Text size="10px" tt="uppercase" lh={1.4} c={empty ? "dimmed" : "cyan"}>
        {label}
      </Text>
      <Text size="xs" fw={600} c={empty ? "dimmed" : undefined} truncate="end">
        {value}
      </Text>
    </Box>
  );

  return (
    <Group
      gap={compact ? "sm" : "lg"}
      wrap="nowrap"
      align="center"
      component="section"
      aria-label="Compute path"
      aria-live="polite"
      data-testid="compute-path"
    >
      {renderNode("01", "Local API", "127.0.0.1")}
      <PathLine active={active} />
      {renderNode("02", "Coordinator", coordinator)}
      <PathLine active={active} />
      {renderNode("03", "Worker", worker, nodes.length < 2)}
      <Badge
        color={statusColors[cluster.status]}
        variant={active ? "filled" : "light"}
        ml="auto"
        leftSection={
          <span
            aria-hidden
            style={{
              display: "inline-block",
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "currentColor",
            }}
          />
        }
      >
        {cluster.status}
      </Badge>
    </Group>
  );
}
