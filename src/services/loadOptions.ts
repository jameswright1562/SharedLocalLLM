import type { ModelLoadOptions } from "../types";

export const DEFAULT_LOAD_OPTIONS: ModelLoadOptions = {
  flashAttention: false,
  useMmap: true,
  useMlock: false,
  cpuThreads: 0,
  batchSize: 512,
};
