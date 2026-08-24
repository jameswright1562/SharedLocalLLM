import { useEffect, useMemo, useState } from "react";
import { Alert, Box, Flex, Group, Progress, Stepper, Text, Title } from "@mantine/core";

import { describeAppError } from "../services/errors";
import type { AppService, AppSnapshot, NetworkBenchmark, NodeCapabilities } from "../types";
import { SetupStepContent } from "./SetupStepContent";

interface SetupWizardProps {
  snapshot: AppSnapshot;
  service: AppService;
  onComplete: (snapshot: AppSnapshot) => void;
}

const steps = ["Runtime", "Device", "Pair", "Models", "Network", "Ready"];

export function SetupWizard({ snapshot, service, onComplete }: SetupWizardProps) {
  const [liveSnapshot, setLiveSnapshot] = useState(snapshot);
  const initialStep = snapshot.runtime.status === "ready" ? 1 : 0;
  const [step, setStep] = useState(initialStep);
  const [deviceName, setDeviceName] = useState(snapshot.deviceName || "Local node");
  const [manualEndpoint, setManualEndpoint] = useState("");
  const [pairedNode, setPairedNode] = useState<NodeCapabilities | null>(snapshot.nodes[1] ?? null);
  const [network, setNetwork] = useState<NetworkBenchmark | undefined>(snapshot.network);
  const [busy, setBusy] = useState(false);
  const [runtimeProgress, setRuntimeProgress] = useState({
    percent: 0,
    status: "Ready to download",
  });
  const [error, setError] = useState("");

  const progress = useMemo(() => ((step + 1) / steps.length) * 100, [step]);

  function clearError() {
    setError("");
  }

  function reportError(reason: unknown, fallback: string) {
    setError(describeAppError(reason, fallback));
  }

  useEffect(() => {
    if (step !== 2 || pairedNode) return;
    let active = true;
    const poll = async () => {
      try {
        const next = await service.getAppSnapshot();
        if (!active) return;
        setLiveSnapshot(next);
        if (next.nodes.length > 1) setPairedNode(next.nodes[1] ?? null);
      } catch {
        /* keep waiting for the code host to persist the peer */
      }
    };
    const timer = window.setInterval(() => void poll(), 2_000);
    void poll();
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [pairedNode, service, step]);

  async function checkAgain() {
    setBusy(true);
    clearError();
    try {
      const next = await service.refreshHardware();
      setLiveSnapshot(next);
      if (next.runtime.status === "ready") setStep(1);
    } catch (reason) {
      reportError(reason, "Hardware and runtime status could not be refreshed.");
    } finally {
      setBusy(false);
    }
  }

  async function connect() {
    setBusy(true);
    clearError();
    try {
      const endpoint = manualEndpoint.trim();
      const node = endpoint ? await service.connectPeer(endpoint) : await service.connectPeer();
      setPairedNode(node);
      setStep(3);
    } catch (reason) {
      reportError(reason, "Could not connect to the other computer.");
    } finally {
      setBusy(false);
    }
  }

  async function installRuntime() {
    setBusy(true);
    clearError();
    try {
      await service.installRuntime((percent, status) => setRuntimeProgress({ percent, status }));
      setRuntimeProgress({ percent: 100, status: "Runtime ready" });
      setStep(1);
    } catch (reason) {
      reportError(reason, "The runtime could not be installed.");
    } finally {
      setBusy(false);
    }
  }

  async function addFolder() {
    setBusy(true);
    clearError();
    try {
      const directory = await service.addModelDirectory();
      if (!directory) {
        setError("No folder selected. Use detected sources or choose another folder.");
        return;
      }
      setStep(4);
    } catch (reason) {
      reportError(reason, "The model folder could not be added.");
    } finally {
      setBusy(false);
    }
  }

  async function testNetwork() {
    setBusy(true);
    clearError();
    try {
      const result = await service.runNetworkTest();
      setNetwork(result);
      setStep(5);
    } catch (reason) {
      reportError(reason, "The network test could not finish.");
    } finally {
      setBusy(false);
    }
  }

  async function finish() {
    setBusy(true);
    clearError();
    try {
      onComplete(await service.completeSetup(deviceName.trim()));
    } catch (reason) {
      reportError(reason, "Setup could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Flex mih="100vh" direction={{ base: "column", md: "row" }}>
      <Box
        component="aside"
        aria-label="Setup progress"
        w={{ base: "100%", md: 340 }}
        p="xl"
        bg="dark.8"
        style={{ borderRight: "1px solid var(--mantine-color-dark-4)" }}
      >
        <Group gap="sm" mb="lg">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
          </div>
          <Text size="xs" fw={700} tt="uppercase" lts={2} c="cyan">
            SharedLocalLLM
          </Text>
        </Group>
        <Title order={1} lh={1}>
          Build your compute link.
        </Title>
        <Text c="dimmed" mt="sm" mb="xl">
          Connect two trusted Windows computers, inspect the link, then load one model across the
          memory they can safely share.
        </Text>
        <Stepper
          orientation="vertical"
          active={step}
          color="cyan"
          size="sm"
          allowNextStepsSelect={false}
          iconSize={30}
        >
          {steps.map((label) => (
            <Stepper.Step key={label} label={label} />
          ))}
        </Stepper>
        <Progress value={progress} mt="xl" color="cyan" size="sm" radius="xs" aria-hidden />
      </Box>
      <Flex
        component="main"
        justify="center"
        align="flex-start"
        p={{ base: "md", md: "xl" }}
        style={{ flex: 1 }}
      >
        <Box w={640} maw="100%">
          <Text size="xs" tt="uppercase" lts={2} c="dimmed" mb="xs">
            Step {step + 1} of {steps.length}
          </Text>
          <SetupStepContent
            step={step}
            snapshot={liveSnapshot}
            service={service}
            deviceName={deviceName}
            setDeviceName={setDeviceName}
            manualEndpoint={manualEndpoint}
            setManualEndpoint={setManualEndpoint}
            pairedNode={pairedNode}
            network={network}
            busy={busy}
            runtimeProgress={runtimeProgress}
            installRuntime={installRuntime}
            checkAgain={checkAgain}
            connect={connect}
            addFolder={addFolder}
            testNetwork={testNetwork}
            setStep={setStep}
            finish={finish}
          />
          {error && (
            <Alert role="alert" variant="light" color="coral" mt="md">
              {error}
            </Alert>
          )}
        </Box>
      </Flex>
    </Flex>
  );
}
