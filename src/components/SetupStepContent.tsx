import type { Dispatch, ReactNode, SetStateAction } from "react";
import {
  Alert,
  Badge,
  Box,
  Button,
  Group,
  Paper,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { IconArrowRight, IconPlus } from "@tabler/icons-react";

import type { AppService, AppSnapshot, NetworkBenchmark, NodeCapabilities } from "../types";
import { PairingPanel } from "./PairingPanel";

interface SetupStepContentProps {
  step: number;
  snapshot: AppSnapshot;
  service: AppService;
  deviceName: string;
  setDeviceName: Dispatch<SetStateAction<string>>;
  manualEndpoint: string;
  setManualEndpoint: Dispatch<SetStateAction<string>>;
  pairedNode: NodeCapabilities | null;
  network?: NetworkBenchmark;
  busy: boolean;
  runtimeProgress: { percent: number; status: string };
  installRuntime: () => Promise<void>;
  checkAgain: () => Promise<void>;
  connect: () => Promise<void>;
  addFolder: () => Promise<void>;
  testNetwork: () => Promise<void>;
  finish: () => Promise<void>;
  setStep: Dispatch<SetStateAction<number>>;
}

export function SetupStepContent(props: SetupStepContentProps) {
  const { step } = props;
  return (
    <>
      {step === 0 && <RuntimeStep {...props} />}
      {step === 1 && <IdentityStep {...props} />}
      {step === 2 && <PairStep {...props} />}
      {step === 3 && <SourcesStep {...props} />}
      {step === 4 && <NetworkStep {...props} />}
      {step === 5 && <ReadyStep {...props} />}
    </>
  );
}

function StepHeading({ kicker, title, lede }: { kicker: string; title: string; lede: string }) {
  return (
    <>
      <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan" mb={4}>
        {kicker}
      </Text>
      <Title order={2}>{title}</Title>
      <Text c="dimmed" mt="xs" mb="lg">
        {lede}
      </Text>
    </>
  );
}

function RuntimeStep({
  snapshot,
  busy,
  runtimeProgress,
  installRuntime,
  checkAgain,
}: SetupStepContentProps) {
  const ready = snapshot.runtime.status === "ready";
  return (
    <section>
      <StepHeading
        kicker="Runtime readiness"
        title="Install the inference runtime"
        lede="SharedLocalLLM uses a verified llama.cpp CUDA runtime. Development tools and LM Studio are not required."
      />
      <Alert
        role="status"
        variant="light"
        color={ready ? "mint" : "amber"}
        title={ready ? "Runtime ready" : "Runtime required"}
        mb="md"
      >
        {ready
          ? (snapshot.runtime.version ?? "The pinned runtime is installed.")
          : "Complete the bundled runtime installation, then check again."}
      </Alert>
      {busy && (
        <Box aria-live="polite" mb="md">
          <Group justify="space-between" mb={4}>
            <Text size="sm">{runtimeProgress.status}</Text>
            <Text size="sm" fw={700}>
              {runtimeProgress.percent}%
            </Text>
          </Group>
          <Progress value={runtimeProgress.percent} animated color="cyan" size="sm" radius="xs" />
        </Box>
      )}
      <Group gap="sm">
        <Button variant="default" disabled={busy} onClick={() => void checkAgain()}>
          Check again
        </Button>
        <Button disabled={busy} onClick={() => void installRuntime()}>
          {busy ? "Installing…" : ready ? "Reinstall runtime" : "Install runtime"}
        </Button>
      </Group>
    </section>
  );
}

function IdentityStep({ snapshot, deviceName, setDeviceName, setStep }: SetupStepContentProps) {
  const valid = deviceName.trim().length > 0 && deviceName.trim().length <= 80;
  return (
    <section>
      <StepHeading
        kicker="Identity"
        title="Name this computer"
        lede="Use a short name you will recognize when choosing a coordinator or reading benchmark results."
      />
      <TextInput
        label="Device name"
        maxLength={80}
        value={deviceName}
        onChange={(event) => setDeviceName(event.target.value)}
        autoFocus
        mb="md"
      />
      <Paper withBorder p="md" bg="dark.8" mb="md">
        <Stack gap={4}>
          <Text size="xs" tt="uppercase" lts={1.5} c="dimmed" fw={600}>
            Detected locally
          </Text>
          <Text fw={600}>{snapshot.nodes[0]?.gpu.name ?? "GPU scan pending"}</Text>
          <Text size="xs" c="dimmed">
            {snapshot.nodes[0]
              ? `${snapshot.nodes[0].ramTotalGb} GB system memory`
              : "Hardware will appear after refresh"}
          </Text>
        </Stack>
      </Paper>
      <Group gap="sm">
        <Button variant="default" onClick={() => setStep(0)}>
          Back
        </Button>
        <Button disabled={!valid} onClick={() => setStep(2)}>
          Continue
        </Button>
      </Group>
    </section>
  );
}

function PairStep({
  manualEndpoint,
  setManualEndpoint,
  pairedNode,
  busy,
  connect,
  setStep,
}: SetupStepContentProps) {
  return (
    <section>
      <StepHeading
        kicker="Peer connection"
        title="Connect the second computer"
        lede="Open SharedLocalLLM on the other computer and connect from either screen. You can finish setup with one computer and connect later from Nodes."
      />
      <PairingPanel
        manualEndpoint={manualEndpoint}
        setManualEndpoint={setManualEndpoint}
        pairedNode={pairedNode}
        busy={busy}
        connect={() => void connect()}
        onContinue={() => setStep(3)}
      />
      <Group gap="sm" mt="md">
        <Button variant="default" onClick={() => setStep(1)}>
          Back
        </Button>
        <Button variant="default" onClick={() => setStep(3)}>
          Skip and use this computer only
        </Button>
      </Group>
    </section>
  );
}

function SourcesStep({ snapshot, addFolder, setStep }: SetupStepContentProps) {
  const lmStudio = snapshot.modelDirectories.some((directory) => directory.source === "lm-studio");
  return (
    <section>
      <StepHeading
        kicker="Model sources"
        title="Choose where models live"
        lede="LM Studio folders are discovered when present. Add any other directory without moving or changing its files. This computer indexes its own files only."
      />
      <Paper
        withBorder
        p="md"
        mb="sm"
        style={{
          borderColor: lmStudio ? "var(--mantine-color-cyan-6)" : undefined,
        }}
      >
        <Group justify="space-between" wrap="nowrap">
          <Group gap="sm" wrap="nowrap">
            <Badge size="lg" variant="light" color="cyan" ff="monospace">
              LM
            </Badge>
            <Box>
              <Text fw={600}>LM Studio models</Text>
              <Text size="sm" c="dimmed">
                Automatic per-computer discovery
              </Text>
            </Box>
          </Group>
          <Badge color={lmStudio ? "cyan" : "gray"} variant="light">
            {lmStudio ? "Detected" : "Not found"}
          </Badge>
        </Group>
      </Paper>
      <Paper
        component="button"
        type="button"
        withBorder
        p="md"
        w="100%"
        onClick={() => void addFolder()}
        style={{ cursor: "pointer", textAlign: "left" }}
      >
        <Group justify="space-between" wrap="nowrap">
          <Group gap="sm" wrap="nowrap">
            <ThemeGlyph glyph={<IconPlus size={16} />} />
            <Box>
              <Text fw={600}>Add a custom folder</Text>
              <Text size="sm" c="dimmed">
                Choose any directory containing GGUF files
              </Text>
            </Box>
          </Group>
          <IconArrowRight size={18} aria-hidden />
        </Group>
      </Paper>
      <Group gap="sm" mt="md">
        <Button variant="default" onClick={() => setStep(2)}>
          Back
        </Button>
        <Button variant="default" onClick={() => setStep(4)}>
          Use detected sources
        </Button>
      </Group>
    </section>
  );
}

function ThemeGlyph({ glyph }: { glyph: ReactNode }) {
  return (
    <Badge size="lg" w={34} h={34} p={0} variant="light" color="cyan" radius="xl">
      {glyph}
    </Badge>
  );
}

function NetworkStep({ network, busy, testNetwork, setStep }: SetupStepContentProps) {
  return (
    <section>
      <StepHeading
        kicker="Link test"
        title="Measure the path between nodes"
        lede="The test sends temporary data over the peer channel. One computer can still run models that fit locally if you skip it."
      />
      <Group justify="center" gap="md" aria-hidden="true" my="xl" wrap="nowrap">
        <Text size="xs" tt="uppercase" lts={2} c="dimmed" fw={600}>
          This PC
        </Text>
        <Box className="link-illustration-line" />
        <Text c="cyan" fw={700}>
          ↔
        </Text>
        <Box className="link-illustration-line" />
        <Text size="xs" tt="uppercase" lts={2} c="dimmed" fw={600}>
          Peer
        </Text>
      </Group>
      {network && (
        <Alert variant="light" color="mint" mb="md">
          Existing result: {Math.round(network.downMbps)} Mbit/s · {network.latencyP95Ms} ms p95
        </Alert>
      )}
      <Group gap="sm">
        <Button variant="default" onClick={() => setStep(3)}>
          Back
        </Button>
        <Button variant="default" onClick={() => setStep(5)}>
          Skip and use this computer only
        </Button>
        <Button disabled={busy} onClick={() => void testNetwork()}>
          {busy ? "Testing link…" : "Run network test"}
        </Button>
      </Group>
    </section>
  );
}

function ReadyStep({
  pairedNode,
  snapshot,
  network,
  busy,
  finish,
  setStep,
}: SetupStepContentProps) {
  return (
    <section>
      <StepHeading
        kicker="Ready"
        title="Your compute link is ready"
        lede="Models will be evaluated against available GPU memory, system memory, and the measured link before launch."
      />
      <SimpleGrid cols={{ base: 1, xs: 3 }} spacing="sm" mb="md">
        {[
          {
            label: "Nodes",
            value: pairedNode || snapshot.nodes.length > 1 ? "2 online" : "1 local",
          },
          { label: "Runtime", value: snapshot.runtime.version ?? "Installed" },
          { label: "Network", value: network?.classification ?? "Not tested", capitalize: true },
        ].map((stat) => (
          <Paper key={stat.label} withBorder p="md" bg="dark.8">
            <Text size="xs" tt="uppercase" lts={1} c="dimmed" fw={600}>
              {stat.label}
            </Text>
            <Text fw={600} tt={stat.capitalize ? "capitalize" : undefined}>
              {stat.value}
            </Text>
          </Paper>
        ))}
      </SimpleGrid>
      <Group gap="sm">
        <Button variant="default" onClick={() => setStep(4)}>
          Back
        </Button>
        <Button disabled={busy} onClick={() => void finish()}>
          {busy ? "Saving setup…" : "Open dashboard"}
        </Button>
      </Group>
    </section>
  );
}
