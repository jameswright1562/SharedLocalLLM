import { useEffect, useRef, useState } from "react";
import {
  Badge,
  Button,
  Group,
  Paper,
  Progress,
  SegmentedControl,
  Stack,
  Text,
} from "@mantine/core";
import type {
  AppService,
  AutotuneStatus,
  ModelLoadConfig,
  ModelRecord,
  ModelTuneResult,
} from "../types";
import { describeAppError } from "../services/errors";

const POLL_MS = 1200;

const depthOptions = [
  { value: "quick", label: "Quick" },
  { value: "full", label: "Full sweep" },
];

export function TunedBadge({ tune }: { tune?: ModelTuneResult }) {
  if (!tune) return null;
  return (
    <Badge color="cyan" variant="light" data-testid="autotune-badge">
      Auto-tuned · {tune.promptTokensPerSecond.toFixed(0)} tok/s prompt
    </Badge>
  );
}

interface AutotunePanelProps {
  model?: ModelRecord;
  tune?: ModelTuneResult;
  service: AppService;
  onMessage: (message: string) => void;
  onRefresh: () => Promise<void>;
  onApplied?: (loadConfig: ModelLoadConfig) => void;
}

export function AutotunePanel({
  model,
  tune,
  service,
  onMessage,
  onRefresh,
  onApplied,
}: AutotunePanelProps) {
  const [depth, setDepth] = useState<"quick" | "full">("quick");
  const [status, setStatus] = useState<AutotuneStatus>({ status: "idle" });
  const [busy, setBusy] = useState(false);
  const running = status.status === "running";
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!running) return undefined;
    pollRef.current = setInterval(() => {
      void service
        .getAutotuneStatus()
        .then((next) => setStatus(next))
        .catch(() => undefined);
    }, POLL_MS);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [running, service]);

  if (!model) {
    return (
      <Paper aria-label="Autotune" p="md">
        <Text c="dimmed">Select a model to tune its launch settings.</Text>
      </Paper>
    );
  }

  const shown = status.result ?? tune;
  const stageProgress = status.stageCount
    ? ((status.stageIndex ?? 0) / status.stageCount) * 100
    : 0;

  async function startTune() {
    setBusy(true);
    try {
      setStatus(await service.startModelAutotune(model!.id, depth));
      onMessage(`Measuring llama.cpp settings for ${model!.name}. This runs real benchmarks.`);
    } catch (reason) {
      setStatus({
        status: "failed",
        error: describeAppError(reason, "The tuning run could not be started."),
      });
    } finally {
      setBusy(false);
    }
  }

  async function cancelTune() {
    setBusy(true);
    try {
      await service.cancelModelAutotune();
      setStatus((current) => ({ ...current, status: "cancelled" }));
    } catch (reason) {
      onMessage(describeAppError(reason, "The tuning run could not be cancelled."));
    } finally {
      setBusy(false);
    }
  }

  async function applyTune() {
    if (!shown) return;
    setBusy(true);
    try {
      const applied = await service.applyModelTune(model!.id);
      onApplied?.(applied.loadConfig);
      onMessage(
        applied.staleTopology
          ? "Tuned settings were saved, but the computers or GPUs changed since this run — consider re-tuning."
          : `Tuned settings saved for ${model!.name}. The next launch uses them.`,
      );
      await onRefresh();
    } catch (reason) {
      onMessage(describeAppError(reason, "The tuned settings could not be applied."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Paper component="section" aria-label="Autotune settings" withBorder p="md">
      <Stack gap="sm">
        <Group justify="space-between" align="center" wrap="nowrap">
          <div>
            <Text size="sm" fw={600}>
              Auto-tune launch settings
            </Text>
            <Text size="xs" c="dimmed">
              Runs the pinned llama-bench against this model and remembers the fastest batch,
              thread, layer, and cache settings. Experimental; measured on your hardware.
            </Text>
          </div>
          <TunedBadge tune={tune} />
        </Group>

        {!running && (
          <Group gap="sm">
            <SegmentedControl
              aria-label="Autotune depth"
              value={depth}
              onChange={(value) => setDepth(value as "quick" | "full")}
              data={depthOptions}
            />
            <Button variant="default" disabled={busy} onClick={() => void startTune()}>
              Start tuning
            </Button>
          </Group>
        )}

        {running && (
          <Stack gap="xs">
            <Progress value={stageProgress} animated color="cyan" aria-label="Autotune progress" />
            <Text size="xs" c="dimmed">
              Stage {status.stageIndex ?? 0} of {status.stageCount}
              {status.currentStage ? ` — ${status.currentStage}` : ""}
            </Text>
            <Button variant="light" color="coral" disabled={busy} onClick={() => void cancelTune()}>
              Stop tuning
            </Button>
          </Stack>
        )}

        {status.status === "failed" && (
          <Text size="sm" c="coral" role="alert">
            {status.error ?? "The tuning run failed."}
          </Text>
        )}
        {status.status === "cancelled" && (
          <Text size="sm" c="dimmed">
            Tuning was stopped before finishing. No settings were changed.
          </Text>
        )}
        {shown && !running && <WinnersSummary tune={shown} />}
        {shown && !running && (
          <Button disabled={busy} onClick={() => void applyTune()}>
            Apply tuned settings
          </Button>
        )}
      </Stack>
    </Paper>
  );
}

function WinnersSummary({ tune }: { tune: ModelTuneResult }) {
  const rows: Array<[string, string]> = [
    ["Batch", tune.winners.batchSize ? `${tune.winners.batchSize}` : "default"],
    ["Micro-batch", tune.winners.uBatch ? `${tune.winners.uBatch}` : "default"],
    ["CPU threads", tune.winners.cpuThreads ? `${tune.winners.cpuThreads}` : "automatic"],
    [
      "KV cache",
      [tune.winners.kvCacheK, tune.winners.kvCacheV].filter(Boolean).join(" / ") || "f16",
    ],
    [
      "GPU layers",
      tune.winners.gpuLayers != null ? `${tune.winners.gpuLayers}` : "placement default",
    ],
  ];
  return (
    <Stack gap={4} data-testid="autotune-winners">
      <Text size="xs" fw={700} tt="uppercase" c="dimmed">
        Winning configuration ({tune.depth})
      </Text>
      {rows.map(([label, value]) => (
        <Text key={label} size="sm">
          {label}:{" "}
          <Text span c="dimmed">
            {value}
          </Text>
        </Text>
      ))}
      <Text size="xs" c="dimmed">
        Measured {tune.promptTokensPerSecond.toFixed(1)} tok/s prompt,{" "}
        {tune.generationTokensPerSecond.toFixed(1)} tok/s generation at{" "}
        {new Date(tune.ranAt).toLocaleString()}.
      </Text>
    </Stack>
  );
}
