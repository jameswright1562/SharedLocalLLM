import { openPath, revealItemInDir } from "@tauri-apps/plugin-opener";

export function modelFolder(modelPath: string | undefined): string | undefined {
  if (!modelPath) return undefined;
  const normalized = modelPath.replaceAll("/", "\\");
  const separator = normalized.lastIndexOf("\\");
  return separator > 0 ? normalized.slice(0, separator) : undefined;
}

export async function openFolderInExplorer(path: string | undefined) {
  if (!path) return;
  await openPath(path);
}

export async function openFileInExplorer(path: string | undefined) {
  if (!path) return;
  await revealItemInDir(path);
}

export async function copyToClipboard(text: string) {
  await navigator.clipboard.writeText(text);
}
