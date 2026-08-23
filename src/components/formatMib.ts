export function formatMib(value: number) {
  return value >= 1024 ? `${(value / 1024).toFixed(2)} GiB` : `${Math.ceil(value)} MiB`;
}
