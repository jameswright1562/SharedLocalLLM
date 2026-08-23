import { useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  DataList,
  Flex,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";

import { PairingPanel } from "../components/PairingPanel";
import { StatusBanner } from "../components/StatusBanner";
import { StatusPill } from "../components/Telemetry";
import { describeAppError } from "../services/errors";
import type { PageProps } from "../types";
import { formatGb } from "./pageFormat";

export function NodesPage({ snapshot, service, refreshSnapshot }: PageProps) {
  const [refreshing, setRefreshing] = useState(false);
  const [confirmingReset, setConfirmingReset] = useState(false);
  const [message, setMessage] = useState("");
  const [manualEndpoint, setManualEndpoint] = useState("");
  async function refresh() {
    setRefreshing(true);
    setMessage("");
    try {
      await service.refreshHardware();
      await refreshSnapshot();
    } catch (reason) {
      setMessage(describeAppError(reason, "Hardware refresh failed."));
    } finally {
      setRefreshing(false);
    }
  }
  async function resetPairing() {
    setRefreshing(true);
    setMessage("");
    try {
      await service.resetPairing();
      await refreshSnapshot();
      setConfirmingReset(false);
      setMessage("Paired node forgotten. Connect again to rejoin the peer.");
    } catch (reason) {
      setMessage(describeAppError(reason, "The paired node could not be forgotten."));
    } finally {
      setRefreshing(false);
    }
  }
  async function connect() {
    setRefreshing(true);
    setMessage("");
    try {
      const endpoint = manualEndpoint.trim();
      await (endpoint ? service.connectPeer(endpoint) : service.connectPeer());
      await refreshSnapshot();
      setMessage("Connected to the other computer.");
    } catch (reason) {
      setMessage(describeAppError(reason, "Could not connect to the other computer."));
    } finally {
      setRefreshing(false);
    }
  }
  return (
    <Box>
      <Flex justify="space-between" align="flex-start" gap="md" wrap="wrap" mb="lg">
        <Box>
          <Text size="xs" fw={700} tt="uppercase" ls={1.5} c="cyan">
            Inventory
          </Text>
          <Title order={1}>Node capabilities</Title>
          <Text c="dimmed">Hardware and availability reported by each trusted computer.</Text>
        </Box>
        <Button variant="default" disabled={refreshing} onClick={() => void refresh()}>
          {refreshing ? "Refreshing…" : "Refresh hardware"}
        </Button>
      </Flex>

      <Stack gap="md">
        {snapshot.nodes.map((node, index) => (
          <Card key={node.id} withBorder p="lg" component="article">
            <Text size="10px" ff="monospace" c="dimmed" tt="uppercase" ls={2} mb={4}>
              Node {String(index + 1).padStart(2, "0")}
            </Text>
            <Group justify="space-between" wrap="nowrap" mb="sm">
              <Box>
                <Title order={3}>{node.name}</Title>
                <Text size="sm" c="dimmed">
                  {node.role} · {node.cpu}
                </Text>
              </Box>
              <StatusPill online={node.online}>{node.online ? "Reachable" : "Offline"}</StatusPill>
            </Group>
            <DataList
              columnCount={{ base: 1, sm: 2 }}
              size="sm"
              mb="md"
              styles={{ itemLabel: { color: "var(--mantine-color-dimmed)", width: 150 } }}
            >
              <DataList.Item>
                <DataList.ItemLabel>Graphics processor</DataList.ItemLabel>
                <DataList.ItemValue>{node.gpu.name}</DataList.ItemValue>
              </DataList.Item>
              <DataList.Item>
                <DataList.ItemLabel>GPU memory</DataList.ItemLabel>
                <DataList.ItemValue>
                  {formatGb(node.gpu.vramAvailableGb)} free / {formatGb(node.gpu.vramTotalGb)}
                </DataList.ItemValue>
              </DataList.Item>
              <DataList.Item>
                <DataList.ItemLabel>System memory</DataList.ItemLabel>
                <DataList.ItemValue>
                  {formatGb(node.ramAvailableGb)} free / {formatGb(node.ramTotalGb)}
                </DataList.ItemValue>
              </DataList.Item>
              <DataList.Item>
                <DataList.ItemLabel>Network path</DataList.ItemLabel>
                <DataList.ItemValue>
                  {node.adapter.name}
                  {node.adapter.linkSpeedMbps ? ` · ${node.adapter.linkSpeedMbps} Mbit/s` : ""}
                </DataList.ItemValue>
              </DataList.Item>
            </DataList>
            {index > 0 && !confirmingReset && (
              <Button
                variant="subtle"
                color="coral"
                size="compact-sm"
                disabled={refreshing}
                onClick={() => setConfirmingReset(true)}
              >
                Forget {node.name}
              </Button>
            )}
            {index > 0 && confirmingReset && (
              <Alert role="alert" variant="light" color="coral" title={`Forget ${node.name}?`}>
                <Stack gap="sm">
                  <Text size="sm">
                    Trust and connection settings will reset. Model files and folders stay
                    untouched.
                  </Text>
                  <Group gap="sm">
                    <Button
                      variant="default"
                      size="xs"
                      disabled={refreshing}
                      onClick={() => setConfirmingReset(false)}
                    >
                      Keep node
                    </Button>
                    <Button
                      color="coral"
                      size="xs"
                      aria-label={`Confirm forget ${node.name}`}
                      disabled={refreshing}
                      onClick={() => void resetPairing()}
                    >
                      {refreshing ? "Forgetting…" : `Forget ${node.name}`}
                    </Button>
                  </Group>
                </Stack>
              </Alert>
            )}
          </Card>
        ))}
        {snapshot.nodes.length < 2 && (
          <>
            <Paper withBorder p="lg">
              <Stack gap={4}>
                <Text size="24px" ff="monospace" c="dimmed" fw={600}>
                  02
                </Text>
                <Title order={3}>No worker connected</Title>
                <Text c="dimmed">
                  The local node can still run models that fit. Enter a peer IP or let discovery
                  find the other computer.
                </Text>
              </Stack>
            </Paper>
            <PairingPanel
              manualEndpoint={manualEndpoint}
              setManualEndpoint={setManualEndpoint}
              pairedNode={null}
              busy={refreshing}
              connect={() => void connect()}
            />
          </>
        )}
      </Stack>

      {message && <StatusBanner message={message} />}
    </Box>
  );
}
