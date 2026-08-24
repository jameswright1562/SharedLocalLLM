import type { ModelRecord } from "../types";
import {
  Badge,
  Box,
  Button,
  NumberFormatter,
  Paper,
  Stack,
  Table,
  Text,
  Title,
} from "@mantine/core";
import { useMemo, useState } from "react";
import {
  IconChevronDown,
  IconChevronUp,
  IconCopy,
  IconFolderOpen,
  IconTrash,
} from "@tabler/icons-react";
import { useContextMenu } from "mantine-contextmenu";
import { formatBytes } from "../pages/pageFormat";
import { copyToClipboard, modelFolder, openFolderInExplorer } from "../helpers/helpers";

export type CapabilityFilter = "all" | "text" | "vision" | "split";

interface ModelCatalogueProps {
  models: ModelRecord[];
  visibleModels: ModelRecord[];
  selectedId: string;
  select: (id: string) => void;
  addFolder: () => void;
  onDelete: (model: ModelRecord) => void;
}

export function ModelCatalogue({
  models,
  visibleModels,
  selectedId,
  select,
  addFolder,
  onDelete,
}: ModelCatalogueProps) {
  const [sortBy, setSortBy] = useState<keyof ModelRecord | null>(null);
  const [reverseSortDirection, setReverseSortDirection] = useState(false);
  const { showContextMenu } = useContextMenu();

  const handleSort = (field: keyof ModelRecord) => {
    const reversed = field === sortBy ? !reverseSortDirection : false;
    setReverseSortDirection(reversed);
    setSortBy(field);
  };

  const sortedModels = useMemo(() => {
    if (!sortBy) return visibleModels;

    return [...visibleModels].sort((a, b) => {
      let comparison: number;

      switch (sortBy) {
        case "name":
          comparison = a.name.localeCompare(b.name);
          break;
        case "architecture":
          comparison = a.architecture.localeCompare(b.architecture);
          break;
        case "sizeBytes":
          comparison = a.sizeBytes - b.sizeBytes;
          break;
        case "capability":
          comparison = a.capability.localeCompare(b.capability);
          break;
        case "contextLength":
          comparison = a.contextLength - b.contextLength;
          break;
        case "locations":
          comparison = a.locations.join(" ").localeCompare(b.locations.join(" "));
          break;
        default:
          return 0;
      }

      return reverseSortDirection ? -comparison : comparison;
    });
  }, [visibleModels, sortBy, reverseSortDirection]);

  if (sortedModels.length === 0) {
    return (
      <Box data-testid="model-list">
        <Paper withBorder p="xl">
          <Stack align="center" gap="xs" ta="center">
            <Badge size="xl" radius="md" variant="light" color="cyan" ff="monospace">
              GG
            </Badge>
            <Title order={3}>{models.length ? "No models match" : "No models indexed"}</Title>
            <Text c="dimmed" maw={420}>
              {models.length
                ? "Clear the search or choose another filter."
                : "Add an LM Studio or custom folder containing GGUF files."}
            </Text>
            <Button variant="default" mt="xs" onClick={addFolder}>
              Add folder
            </Button>
          </Stack>
        </Paper>
      </Box>
    );
  }

  return (
    <Box data-testid="model-list">
      <Table.ScrollContainer minWidth={560} h={420}>
        <Table stickyHeader stickyHeaderOffset={0} highlightOnHover verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <SortableTh
                label="Name"
                field="name"
                sortBy={sortBy}
                reverseSortDirection={reverseSortDirection}
                onSort={handleSort}
              />
              <SortableTh
                label="Architecture"
                field="architecture"
                sortBy={sortBy}
                reverseSortDirection={reverseSortDirection}
                onSort={handleSort}
              />
              <SortableTh
                label="Modalities"
                field="capability"
                sortBy={sortBy}
                reverseSortDirection={reverseSortDirection}
                onSort={handleSort}
              />
              <SortableTh
                label="Max Context Size"
                field="contextLength"
                sortBy={sortBy}
                reverseSortDirection={reverseSortDirection}
                onSort={handleSort}
              />
              <SortableTh
                label="Model Size (GB)"
                field="sizeBytes"
                sortBy={sortBy}
                reverseSortDirection={reverseSortDirection}
                onSort={handleSort}
              />
              <SortableTh
                label="Location"
                field="locations"
                sortBy={sortBy}
                reverseSortDirection={reverseSortDirection}
                onSort={handleSort}
              />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sortedModels.map((model) => (
              <Table.Tr
                key={model.id}
                className={selectedId === model.id ? "model-row-selected" : undefined}
                onClick={() => select(model.id)}
                onContextMenu={showContextMenu([
                  {
                    key: "copy-id",
                    icon: <IconCopy size={16} />,
                    title: "Copy ID",
                    onClick: () => void copyToClipboard(model.id),
                  },
                  {
                    key: "open-folder",
                    icon: <IconFolderOpen size={16} />,
                    title: "Open folder",
                    onClick: () => void openFolderInExplorer(modelFolder(model.locations[0]?.path)),
                  },
                  ...(model.isLocal
                    ? [
                        {
                          key: "delete-folder",
                          icon: <IconTrash size={16} />,
                          title: "Delete folder…",
                          color: "red",
                          onClick: () => onDelete(model),
                        },
                      ]
                    : []),
                ])}
                aria-pressed={selectedId === model.id}
                style={{ cursor: "pointer" }}
              >
                <Table.Td>{model.name}</Table.Td>
                <Table.Td>{model.architecture}</Table.Td>
                <Table.Td>
                  <Badge variant="light" size="sm" tt="lowercase">
                    {model.capability}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  <NumberFormatter value={model.contextLength} thousandSeparator />
                </Table.Td>
                <Table.Td>{formatBytes(model.sizeBytes)}</Table.Td>
                <Table.Td>{model.locations.map((location) => location.path).join(", ")}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Box>
  );
}

function SortableTh({
  label,
  field,
  sortBy,
  reverseSortDirection,
  onSort,
}: {
  label: string;
  field: keyof ModelRecord;
  sortBy: keyof ModelRecord | null;
  reverseSortDirection: boolean;
  onSort: (field: keyof ModelRecord) => void;
}) {
  const active = sortBy === field;
  return (
    <Table.Th onClick={() => onSort(field)} style={{ cursor: "pointer", whiteSpace: "nowrap" }}>
      {label}
      {active &&
        (reverseSortDirection ? <IconChevronUp size={14} /> : <IconChevronDown size={14} />)}
    </Table.Th>
  );
}
