import { Alert } from "@mantine/core";

interface StatusBannerProps {
  message: string;
  role?: "status" | "alert";
}

export function StatusBanner({ message, role = "status" }: StatusBannerProps) {
  return (
    <Alert
      role={role}
      data-testid="toast-message"
      variant="light"
      color={role === "alert" ? "coral" : "cyan"}
      mt="md"
    >
      {message}
    </Alert>
  );
}
