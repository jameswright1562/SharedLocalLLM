export class AppError extends Error {
  readonly code?: string;
  readonly action?: string;

  constructor(message: string, options: { code?: string; action?: string } = {}) {
    super(message);
    this.name = "AppError";
    this.code = options.code;
    this.action = options.action;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function decodeAppError(reason: unknown): AppError {
  if (reason instanceof AppError) return reason;
  if (reason instanceof Error) return new AppError(reason.message);
  if (typeof reason === "string" && reason.trim()) return new AppError(reason);
  if (isRecord(reason) && typeof reason.message === "string") {
    return new AppError(reason.message, {
      code: typeof reason.code === "string" ? reason.code : undefined,
      action: typeof reason.action === "string" ? reason.action : undefined,
    });
  }
  return new AppError("An unexpected application error occurred.");
}

export function describeAppError(reason: unknown, fallback?: string): string {
  const error = decodeAppError(reason);
  const message = error.message || fallback || "An unexpected application error occurred.";
  return error.action ? `${message} ${error.action}` : message;
}
