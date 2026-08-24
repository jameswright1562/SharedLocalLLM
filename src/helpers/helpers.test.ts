import { afterEach, describe, expect, it, vi } from "vitest";

const { openPath, revealItemInDir } = vi.hoisted(() => ({
  openPath: vi.fn(),
  revealItemInDir: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-opener", () => ({ openPath, revealItemInDir }));

import { copyToClipboard, modelFolder, openFileInExplorer, openFolderInExplorer } from "./helpers";

describe("helpers", () => {
  afterEach(() => {
    openPath.mockClear();
    revealItemInDir.mockClear();
    vi.restoreAllMocks();
  });

  it("resolves the folder that contains a model file", () => {
    expect(modelFolder("C:\\Models\\hub\\Muse-30B\\muse.gguf")).toBe("C:\\Models\\hub\\Muse-30B");
    expect(modelFolder("C:/Users/James/.lmstudio/models/m.gguf")).toBe(
      "C:\\Users\\James\\.lmstudio\\models",
    );
    expect(modelFolder("relative\\model.gguf")).toBe("relative");
  });

  it("returns no folder for missing or root-only paths", () => {
    expect(modelFolder(undefined)).toBeUndefined();
    expect(modelFolder("")).toBeUndefined();
    expect(modelFolder("model.gguf")).toBeUndefined();
  });

  it("opens folders and files through the native opener", async () => {
    await openFolderInExplorer("C:\\Models");
    await openFileInExplorer("C:\\Models\\m.gguf");
    await openFolderInExplorer(undefined);
    await openFileInExplorer(undefined);
    expect(openPath).toHaveBeenCalledWith("C:\\Models");
    expect(revealItemInDir).toHaveBeenCalledWith("C:\\Models\\m.gguf");
  });

  it("copies text to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    await copyToClipboard("model-id");
    expect(writeText).toHaveBeenCalledWith("model-id");
  });
});
