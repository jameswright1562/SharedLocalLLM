import { Badge, Progress } from "@mantine/core";

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
  return <Progress value={percent} color={tone} size="sm" radius="xs" aria-hidden />;
}

export function StatusPill({ online, children }: { online: boolean; children: string }) {
  return (
    <Badge
      color={online ? "mint" : "dark"}
      variant={online ? "light" : "default"}
      leftSection={
        <span
          aria-hidden
          style={{
            display: "inline-block",
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: online ? "var(--mantine-color-mint-4)" : "var(--mantine-color-dark-3)",
          }}
        />
      }
    >
      {children}
    </Badge>
  );
}
