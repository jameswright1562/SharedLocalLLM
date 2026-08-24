import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { render } from "../test/render";
import { cloneSnapshot } from "../test/fixtures";
import type { ModelRecord } from "../types";
import { ModelCatalogue } from "./ModelCatalogue";

function setup(models: ModelRecord[], visibleModels: ModelRecord[], onDelete = vi.fn()) {
  const select = vi.fn();
  const addFolder = vi.fn();
  render(
    <ModelCatalogue
      models={models}
      visibleModels={visibleModels}
      selectedId=""
      select={select}
      addFolder={addFolder}
      onDelete={onDelete}
    />,
  );
  return { select, addFolder, onDelete };
}

function rowTexts() {
  return within(screen.getByTestId("model-list"))
    .getAllByRole("row")
    .slice(1)
    .map((row) => row.textContent);
}

async function openContextMenu(row: HTMLElement) {
  fireEvent.contextMenu(row);
  await screen.findByText("Copy ID");
}

describe("ModelCatalogue", () => {
  it("renders model rows and reports a selection", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    const { select } = setup(snapshot.models, snapshot.models);
    await user.click(screen.getByText(/orchid 9b/i).closest("tr") as HTMLElement);
    expect(select).toHaveBeenCalledWith("model-text");
  });

  it("sorts by name and reverses on a second click", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    setup(snapshot.models, snapshot.models);

    await user.click(screen.getByRole("columnheader", { name: /name/i }));
    expect(rowTexts()[0]).toContain("Atlas Vision 12B");

    await user.click(screen.getByRole("columnheader", { name: /name/i }));
    expect(rowTexts()[0]).toContain("Orchid 9B Q4_K_M");
  });

  it("sorts by context length and location", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    setup(snapshot.models, snapshot.models);

    await user.click(screen.getByRole("columnheader", { name: /max context size/i }));
    expect(rowTexts()[0]).toContain("Atlas Vision 12B");

    await user.click(screen.getByRole("columnheader", { name: /location/i }));
    expect(rowTexts()[0]).toContain("Orchid 9B Q4_K_M");
  });

  it("offers copy, open folder, and delete actions for a local model", async () => {
    const user = userEvent.setup();
    const snapshot = cloneSnapshot();
    const { onDelete } = setup(snapshot.models, snapshot.models);

    const orchidRow = screen.getByText(/orchid 9b/i).closest("tr") as HTMLElement;
    await openContextMenu(orchidRow);
    await user.click(screen.getByText("Copy ID"));
    await openContextMenu(orchidRow);
    await user.click(screen.getByText("Delete folder…"));

    expect(onDelete).toHaveBeenCalledWith(snapshot.models[0]);
  });

  it("hides deletion for models stored only on the paired computer", async () => {
    const snapshot = cloneSnapshot();
    const remote = snapshot.models.filter((model) => model.id === "model-vision");
    setup(remote, remote);

    await openContextMenu(screen.getByText(/atlas vision/i).closest("tr") as HTMLElement);

    expect(screen.queryByText(/delete folder/i)).not.toBeInTheDocument();
  });

  it("shows the no-models-match state for a filtered-out list", async () => {
    const snapshot = cloneSnapshot();
    const { addFolder } = setup(snapshot.models, []);
    expect(screen.getByText(/no models match/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /add folder/i }));
    expect(addFolder).toHaveBeenCalled();
  });

  it("shows the empty index state and offers to add a folder", async () => {
    const { addFolder } = setup([], []);
    expect(screen.getByText(/no models indexed/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /add folder/i }));
    expect(addFolder).toHaveBeenCalled();
  });
});
