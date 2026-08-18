import type { ModelRecord } from "../types";
import { NumberFormatter, Table } from "@mantine/core";
import { useMemo, useState } from "react";
import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";

export type CapabilityFilter = "all" | "text" | "vision" | "split";

interface ModelCatalogueProps {
  models: ModelRecord[];
  visibleModels: ModelRecord[];
  selectedId: string;
  select: (id: string) => void;
  addFolder: () => void;
}

export function ModelCatalogue({
  models,
  visibleModels,
  selectedId,
  select,
  addFolder,
}: ModelCatalogueProps) {
  const [sortBy, setSortBy] = useState<keyof ModelRecord | null>(null);
  const [reverseSortDirection, setReverseSortDirection] = useState(false);

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

  return (
    <div className="model-list" data-testid="model-list">
      {sortedModels.length === 0 ? (
        <div className="empty-state model-empty">
          <span>GG</span>
          <div>
            <h2>{models.length ? "No models match" : "No models indexed"}</h2>
            <p>
              {models.length
                ? "Clear the search or choose another filter."
                : "Add an LM Studio or custom folder containing GGUF files."}
            </p>
            <button className="button secondary" onClick={addFolder}>
              Add folder
            </button>
          </div>
        </div>
      ) : (
        <Table.ScrollContainer minWidth={500} maxHeight={700}>
          <Table stickyHeader stickyHeaderOffset={0} highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th onClick={() => handleSort("name")}>
                  Name
                  {sortBy === "name" &&
                    (reverseSortDirection ? (
                      <IconChevronUp size={14} />
                    ) : (
                      <IconChevronDown size={14} />
                    ))}
                </Table.Th>
                <Table.Th onClick={() => handleSort("architecture")}>
                  Architecture
                  {sortBy === "architecture" &&
                    (reverseSortDirection ? (
                      <IconChevronUp size={14} />
                    ) : (
                      <IconChevronDown size={14} />
                    ))}
                </Table.Th>
                <Table.Th onClick={() => handleSort("capability")}>
                  Modalities
                  {sortBy === "capability" &&
                    (reverseSortDirection ? (
                      <IconChevronUp size={14} />
                    ) : (
                      <IconChevronDown size={14} />
                    ))}
                </Table.Th>
                <Table.Th onClick={() => handleSort("contextLength")}>
                  Max Context Size
                  {sortBy === "contextLength" &&
                    (reverseSortDirection ? (
                      <IconChevronUp size={14} />
                    ) : (
                      <IconChevronDown size={14} />
                    ))}
                </Table.Th>
                <Table.Th onClick={() => handleSort("locations")}>
                  Location
                  {sortBy === "locations" &&
                    (reverseSortDirection ? (
                      <IconChevronUp size={14} />
                    ) : (
                      <IconChevronDown size={14} />
                    ))}
                </Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {sortedModels.map((model) => (
                <Table.Tr
                  className={`${selectedId === model.id ? "selected" : ""}`}
                  key={model.id}
                  onClick={() => select(model.id)}
                  aria-pressed={selectedId === model.id}
                >
                  <Table.Td>{model.name}</Table.Td>
                  <Table.Td>{model.architecture}</Table.Td>
                  <Table.Td>
                    <i>{model.capability}</i>
                  </Table.Td>
                  <Table.Td>
                    <NumberFormatter value={model.contextLength} thousandSeparator />
                  </Table.Td>
                  <Table.Td>{model.locations.map((location) => location.path).join(", ")}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      )}
    </div>
  );
}
