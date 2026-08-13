import type { ModelFit } from "../types";

export function formatGb(value: number) {
  return `${value.toFixed(value < 10 ? 1 : 0)} GB`;
}

export function formatBytes(value: number) {
  return `${(value / 1_000_000_000).toFixed(1)} GB`;
}

export function formatContext(value: number) {
  return value >= 1000 ? `${Math.round(value / 1000)}k` : String(value);
}

export const fitLabels: Record<ModelFit, string> = {
  "single-node": "Single node",
  "combined-gpu": "Combined GPU",
  "gpu-ram": "GPU + RAM",
  "does-not-fit": "Does not fit",
};
