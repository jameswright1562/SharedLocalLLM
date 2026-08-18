import type { ModelLoadOptions } from "../types";

export function AdvancedLoadOptions({
  options,
  setOptions,
}: {
  options: ModelLoadOptions;
  setOptions: (options: ModelLoadOptions) => void;
}) {
  return (
    <section className="load-options" aria-label="Advanced load options">
      <h3>Advanced load options</h3>
      <ToggleOption
        label="Flash attention"
        checked={options.flashAttention}
        onChange={(checked) => setOptions({ ...options, flashAttention: checked })}
        description="Uses Flash Attention to accelerate generation and shrink the KV cache on compatible GPUs."
        warning={
          options.flashAttention
            ? "Unsupported GPUs fall back to standard attention, so compare speed after enabling."
            : undefined
        }
      />
      <ToggleOption
        label="Memory-map model file"
        checked={options.useMmap}
        onChange={(checked) => setOptions({ ...options, useMmap: checked })}
        description="Maps the model file into memory, using less RAM and starting faster."
        warning={
          options.useMmap
            ? "The model file must stay in place and unlocked while the model is loaded."
            : "Disabling mmap copies the weights into RAM and locks the file for the session."
        }
      />
      <ToggleOption
        label="Lock model in RAM"
        checked={options.useMlock}
        onChange={(checked) => setOptions({ ...options, useMlock: checked })}
        description="Prevents Windows from swapping model memory to disk for steadier performance."
        warning={
          options.useMlock
            ? "Needs free RAM equal to the whole model; launch can fail when memory is tight."
            : undefined
        }
      />
      <NumberOption
        label="CPU threads"
        value={options.cpuThreads}
        min={0}
        onChange={(value) => setOptions({ ...options, cpuThreads: value })}
        hint="0 = automatic"
        description="Threads used for prompt and batch compute. 0 lets llama.cpp choose."
        warning={
          options.cpuThreads > 0
            ? "Values above your physical core count can oversubscribe the CPU and slow generation."
            : undefined
        }
      />
      <NumberOption
        label="Batch size"
        value={options.batchSize}
        min={1}
        onChange={(value) => setOptions({ ...options, batchSize: value })}
        hint="tokens"
        description="Tokens processed per batch during prompt evaluation."
        warning={
          options.batchSize > 1024
            ? "Very large batches use more memory and can exceed available RAM during prompt evaluation."
            : undefined
        }
      />
    </section>
  );
}

function ToggleOption({
  label,
  checked,
  onChange,
  description,
  warning,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  description: string;
  warning?: string;
}) {
  return (
    <div className="load-option">
      <label className="load-option-toggle">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>{label}</span>
      </label>
      <p className="option-description">{description}</p>
      {warning && <p className="option-warning">{warning}</p>}
    </div>
  );
}

function NumberOption({
  label,
  value,
  min,
  onChange,
  hint,
  description,
  warning,
}: {
  label: string;
  value: number;
  min: number;
  onChange: (value: number) => void;
  hint: string;
  description: string;
  warning?: string;
}) {
  const id = `load-option-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  return (
    <div className="load-option">
      <div className="load-option-toggle">
        <label htmlFor={id}>{label}</label>
        <span className="option-hint">{hint}</span>
      </div>
      <div className="option-number-row">
        <input
          type="number"
          id={id}
          min={min}
          value={value}
          onChange={(event) => {
            const next = Number.parseInt(event.target.value, 10);
            onChange(Number.isFinite(next) ? next : min);
          }}
        />
        <p className="option-description">{description}</p>
      </div>
      {warning && <p className="option-warning">{warning}</p>}
    </div>
  );
}
