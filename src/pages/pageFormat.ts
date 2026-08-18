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

export function formatRunTime(ranAt?: string | null): string {
  if (!ranAt) return "—";
  const trimmed = ranAt.trim();
  if (!trimmed) return "—";
  const date = /^\d+$/.test(trimmed) ? new Date(Number(trimmed) * 1000) : new Date(trimmed);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export const fitLabels: Record<ModelFit, string> = {
  "single-node": "Single node",
  "combined-gpu": "Combined GPU",
  "gpu-ram": "GPU + RAM",
  "does-not-fit": "Does not fit",
};
