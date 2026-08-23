import { Alert, Badge, Button, Group, Paper, Text, TextInput, Title } from "@mantine/core";

import type { NodeCapabilities } from "../types";

interface PairingPanelProps {
  manualEndpoint: string;
  setManualEndpoint: (value: string) => void;
  pairedNode: NodeCapabilities | null;
  busy: boolean;
  connect: () => void;
  onContinue?: () => void;
}

export function PairingPanel({
  manualEndpoint,
  setManualEndpoint,
  pairedNode,
  busy,
  connect,
  onContinue,
}: PairingPanelProps) {
  return (
    <>
      <Paper withBorder p="lg">
        <Group gap="sm" mb="xs" wrap="nowrap">
          <Badge size="lg" w={34} h={34} p={0} radius="xl" variant="light" color="cyan">
            A
          </Badge>
          <Title order={3}>Connect to the second computer</Title>
        </Group>
        <Text c="dimmed" mb="md">
          Both computers auto-discover each other. For a direct cable, enter the peer&apos;s IPv4
          address.
        </Text>
        <TextInput
          id="manual-peer-endpoint"
          label="Ethernet IPv4 address (optional)"
          inputMode="decimal"
          placeholder="10.10.10.2"
          value={manualEndpoint}
          onChange={(event) => setManualEndpoint(event.target.value)}
          mb={4}
        />
        <Text size="xs" c="dimmed" mb="md">
          Port 49158 is automatic.
        </Text>
        <Button disabled={busy} onClick={connect}>
          {busy ? "Connecting…" : "Connect"}
        </Button>
      </Paper>
      {pairedNode && (
        <Alert variant="light" color="mint" mt="md">
          Paired with {pairedNode.name}
        </Alert>
      )}
      {pairedNode && onContinue && (
        <Group mt="md">
          <Button onClick={onContinue}>Continue</Button>
        </Group>
      )}
    </>
  );
}
