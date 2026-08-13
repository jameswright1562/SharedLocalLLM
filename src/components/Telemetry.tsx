export function Meter({
  value,
  max,
  tone = "cyan",
}: {
  value: number;
  max: number;
  tone?: "cyan" | "amber";
}) {
  const percent = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <span className={`meter ${tone}`} aria-hidden="true">
      <i style={{ width: `${percent}%` }} />
    </span>
  );
}

export function StatusPill({ online, children }: { online: boolean; children: string }) {
  return (
    <span className={`status-pill ${online ? "online" : "offline"}`}>
      <i aria-hidden="true" />
      {children}
    </span>
  );
}
