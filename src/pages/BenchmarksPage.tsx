import { useRef, useState } from "react";
import {
  Alert,
  Badge,
  Box,
  Button,
  Flex,
  Group,
  NativeSelect,
  Paper,
  Stack,
  Table,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { IconInfoCircle, IconPlayerPlay } from "@tabler/icons-react";

import { describeAppError } from "../services/errors";
import { fitLayersByVram } from "../services/splitEstimate";
import type { InferenceBenchmark, PageProps } from "../types";
import { formatRunTime } from "./pageFormat";

export function BenchmarksPage({ snapshot, service, refreshSnapshot, navigate }: PageProps) {
  const [modelId, setModelId] = useState(snapshot.models[0]?.id ?? "");
  const [runs, setRuns] = useState<InferenceBenchmark[]>(snapshot.benchmarks);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const runRef = useRef(0);
  const selectedModel = snapshot.models.find((model) => model.id === modelId);
  const gpuNodes = snapshot.nodes.filter((node) => node.online && node.gpu.vramAvailableGb > 0);
  const plannedSplit = selectedModel ? fitLayersByVram(selectedModel, gpuNodes) : [];
  const activeModelId =
    snapshot.cluster.status === "running"
      ? snapshot.cluster.modelId
      : snapshot.nodes.find((node) => node.clusterStatus === "running")?.clusterModelId;
  const usesRunningInstance = Boolean(activeModelId && modelId === activeModelId);

  async function runBenchmark() {
    if (!modelId || running) return;
    const runId = ++runRef.current;
    setRunning(true);
    setError("");
    try {
      const results = await service.runInferenceBenchmark(modelId);
      if (runRef.current !== runId) return;
      setRuns((current) => [...results, ...current]);
      await refreshSnapshot();
    } catch (reason) {
      if (runRef.current === runId) {
        const detail = describeAppError(reason, "The benchmark could not finish.");
        setError(detail);
        setRuns((current) => [
          {
            id: `failed-${Date.now()}`,
            modelName: selectedModel?.name ?? modelId,
            topology: "local",
            promptTokensPerSecond: 0,
            generationTokensPerSecond: 0,
            loadTimeSeconds: 0,
            memoryPeakGb: 0,
            recommended: false,
            ranAt: new Date().toISOString(),
            error: detail,
          },
          ...current,
        ]);
      }
    } finally {
      if (runRef.current === runId) setRunning(false);
    }
  }

  async function cancelBenchmark() {
    runRef.current += 1;
    setRunning(false);
    setError("");
    try {
      await service.cancelInferenceBenchmark();
    } catch (reason) {
      setError(describeAppError(reason, "The benchmark could not be cancelled."));
    }
  }

  return (
    <Box>
      <Flex justify="space-between" align="flex-start" gap="md" wrap="wrap" mb="lg">
        <Box>
          <Text size="xs" fw={700} tt="uppercase" ls={1.5} c="cyan">
            Placement evidence
          </Text>
          <Title order={1}>Performance benchmarks</Title>
          <Text c="dimmed">
            Measured results for this exact model, hardware pair, context, and network route.
          </Text>
        </Box>
        <Box w={320} maw="100%">
          <NativeSelect
            id="benchmark-model"
            label="Benchmark model"
            value={modelId}
            disabled={running || snapshot.models.length === 0}
            onChange={(event) => setModelId(event.target.value)}
            data={snapshot.models.map((model) => ({ value: model.id, label: model.name }))}
            mb="xs"
          />
          {usesRunningInstance ? (
            <Alert role="status" variant="light" color="cyan" p="xs">
              <Text size="xs">
                {selectedModel?.name} is running — benchmarks the loaded instance without
                reloading.
              </Text>
            </Alert>
          ) : (
            plannedSplit.length > 0 && (
              <Alert role="status" variant="light" color="cyan" p="xs">
                <Text size="xs">
                  Automatic GPU split:{" "}
                  {plannedSplit
                    .map((allocation) => {
                      const name = gpuNodes.find((node) => node.id === allocation.nodeId)?.name;
                      return `${name ?? "Unknown node"}: ${allocation.layers} layers`;
                    })
                    .join(" · ")}
                </Text>
              </Alert>
            )
          )}
        </Box>
      </Flex>

      <Group gap="sm" mb="lg">
        {running ? (
          <Button color="coral" variant="light" onClick={() => void cancelBenchmark()}>
            Cancel benchmark
          </Button>
        ) : (
          <Button disabled={!modelId} onClick={() => void runBenchmark()}>
            Run benchmark
          </Button>
        )}
        <Button variant="subtle" onClick={() => navigate("models")}>
          Benchmark a model
        </Button>
      </Group>

      {error && (
        <Alert role="alert" variant="light" color="coral" mb="md">
          {error}
        </Alert>
      )}

      {runs.length === 0 ? (
        <Paper withBorder p="xl">
          <Stack align="center" gap="xs" ta="center">
            <ThemeIcon variant="light" color="cyan" size="xl" radius="xl">
              <IconPlayerPlay size={20} />
            </ThemeIcon>
            <Title order={3}>No benchmark runs</Title>
            <Text c="dimmed" maw={420}>
              Choose a model to compare valid single-node and distributed placements.
            </Text>
          </Stack>
        </Paper>
      ) : (
        <Table.ScrollContainer minWidth={720} mb="md">
          <Table verticalSpacing="sm" horizontalSpacing="md">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Model / topology</Table.Th>
                <Table.Th>Prompt</Table.Th>
                <Table.Th>Generation</Table.Th>
                <Table.Th>Duration</Table.Th>
                <Table.Th>Peak memory</Table.Th>
                <Table.Th>Run</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {runs.map((benchmark) => (
                <Table.Tr key={benchmark.id} data-recommended={benchmark.recommended || undefined}>
                  <Table.Td>
                    <Stack gap={0}>
                      <Text fw={600}>{benchmark.modelName}</Text>
                      <Group gap={6} align="center">
                        <Text size="xs" c="dimmed" tt="capitalize">
                          {benchmark.error ? `Failed · ${benchmark.error}` : ""}
                          {benchmark.topology}
                          {benchmark.gpuLayers?.length
                            ? ` · ${benchmark.gpuLayers.map((item) => item.layers).join("/")} GPU layers`
                            : ""}
                        </Text>
                        {benchmark.recommended && (
                          <Badge variant="light" color="mint" size="xs">
                            Recommended
                          </Badge>
                        )}
                      </Group>
                    </Stack>
                  </Table.Td>
                  <ThroughputCell value={benchmark.promptTokensPerSecond} />
                  <ThroughputCell value={benchmark.generationTokensPerSecond} />
                  <Table.Td>{benchmark.loadTimeSeconds.toFixed(1)} s</Table.Td>
                  <Table.Td>
                    {benchmark.memoryPeakGb > 0 ? `${benchmark.memoryPeakGb.toFixed(1)} GB` : "—"}
                  </Table.Td>
                  <Table.Td>{formatRunTime(benchmark.ranAt)}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      )}

      <Paper withBorder p="md" bg="dark.8">
        <Group gap="sm" wrap="nowrap" align="flex-start">
          <ThemeIcon variant="light" color="amber" size="md" mt={2}>
            <IconInfoCircle size={14} />
          </ThemeIcon>
          <Text size="sm" c="dimmed">
            <b>Why results differ</b> Prompt processing and token generation stress the link
            differently. SharedLocalLLM recommends the fastest valid result; distribution is not
            assumed to be faster.
          </Text>
        </Group>
      </Paper>
    </Box>
  );
}

function ThroughputCell({ value }: { value: number }) {
  return (
    <Table.Td>
      <Stack gap={0}>
        <Text fw={600}>{value.toFixed(1)}</Text>
        <Text size="xs" c="dimmed">
          tok/s
        </Text>
      </Stack>
    </Table.Td>
  );
}
