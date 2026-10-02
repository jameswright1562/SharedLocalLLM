import { Group, Paper, Progress, Stack, Text } from "@mantine/core";
import type { ClusterSession } from "../types";
import { formatBytes } from "../pages/pageFormat";
import { loadStagesFor, usesWorker } from "../services/loadStages";

interface LoadProgressProps {
  cluster: ClusterSession;
  workerName?: string;
}

export function LoadProgress({ cluster, workerName }: LoadProgressProps) {
  if (cluster.status !== "loading") return null;
  const stages = loadStagesFor(cluster);
  const stageIndex = stages.findIndex((stage) => stage.id === cluster.stage);
  const stageLabel = stages[stageIndex]?.label ?? "starting";
  const sent = cluster.bytesToWorker;
  const expected = cluster.expectedBytes;
  const hasEstimate = typeof sent === "number" && typeof expected === "number" && expected > 0;
  const percent = hasEstimate ? Math.max(0, Math.min(100, (sent / expected) * 100)) : 0;
  const target = workerName ?? "the worker";

  return (
    <Paper
      withBorder
      p="md"
      mt="md"
      component="section"
      aria-label="Model load progress"
      role="status"
      data-testid="load-progress"
    >
      <Stack gap="sm">
        <Text size="sm" fw={600}>
          Loading{cluster.modelId ? ` ${cluster.modelId}` : " model"} — {stageLabel}
        </Text>
        <Stack gap={4}>
          {stages.map((stage, index) => {
            const done = stageIndex >= 0 && index < stageIndex;
            const current = index === stageIndex;
            return (
              <Group key={stage.id} gap="xs" wrap="nowrap">
                <Text size="xs" fw={700} c={done ? "mint" : current ? "cyan" : "dimmed"} w={18}>
                  {done ? "✓" : `${index + 1}`}
                </Text>
                <Text size="xs" c={done || current ? undefined : "dimmed"} fw={current ? 600 : 400}>
                  {stage.label}
                </Text>
              </Group>
            );
          })}
        </Stack>
        {hasEstimate ? (
          <Stack gap={4}>
            <Progress value={percent} color="cyan" aria-label="Transfer progress" />
            <Text size="xs" c="dimmed">
              ≈ {formatBytes(sent)} of {formatBytes(expected)} sent to {target} · estimate, includes
              protocol overhead
            </Text>
          </Stack>
        ) : (
          <Stack gap={4}>
            <Progress value={100} striped animated color="cyan" aria-label="Loading" />
            <Text size="xs" c="dimmed">
              {usesWorker(cluster)
                ? `Worker transfer to ${target} — measuring…`
                : `${stageLabel} — waiting for the model to become ready…`}
            </Text>
          </Stack>
        )}
      </Stack>
    </Paper>
  );
}
