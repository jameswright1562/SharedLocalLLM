import type { GpuLayerAllocation, ModelLoadConfig, ModelLoadOptions } from "../types";
import { DEFAULT_LOAD_OPTIONS } from "./loadOptions";

type SavedLoadConfigs = Record<string, ModelLoadConfig>;

export function savedContextSizes(configs: SavedLoadConfigs): Record<string, number> {
  return pickValues(configs, (config) =>
    typeof config.contextSize === "number" ? config.contextSize : undefined,
  );
}

export function savedManualSplits(configs: SavedLoadConfigs): Record<string, boolean> {
  return pickValues(configs, (config) => ((config.gpuLayers?.length ?? 0) > 0 ? true : undefined));
}

export function savedLayerSplits(configs: SavedLoadConfigs): Record<string, GpuLayerAllocation[]> {
  return pickValues(configs, (config) => (config.gpuLayers?.length ? config.gpuLayers : undefined));
}

export function savedRemoteCpuFlags(configs: SavedLoadConfigs): Record<string, boolean> {
  return pickValues(configs, (config) => (config.includeRemoteCpu ? true : undefined));
}

export function savedForceLaunches(configs: SavedLoadConfigs): Record<string, boolean> {
  return pickValues(configs, (config) => (config.force ? true : undefined));
}

export function savedOptionValues(configs: SavedLoadConfigs): Record<string, ModelLoadOptions> {
  return pickValues(configs, (config) => ({
    flashAttention: Boolean(config.flashAttention),
    useMmap: config.useMmap ?? DEFAULT_LOAD_OPTIONS.useMmap,
    useMlock: Boolean(config.useMlock),
    cpuThreads: config.cpuThreads ?? DEFAULT_LOAD_OPTIONS.cpuThreads,
    batchSize: config.batchSize ?? DEFAULT_LOAD_OPTIONS.batchSize,
  }));
}

function pickValues<T>(
  configs: SavedLoadConfigs,
  map: (config: ModelLoadConfig) => T | undefined,
): Record<string, T> {
  const values: Record<string, T> = {};
  for (const [id, config] of Object.entries(configs)) {
    const value = map(config);
    if (value !== undefined) values[id] = value;
  }
  return values;
}
