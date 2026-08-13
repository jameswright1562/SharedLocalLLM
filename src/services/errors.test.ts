import { describe, expect, it } from "vitest";

import { decodeAppError } from "./errors";

describe("decodeAppError", () => {
  it("preserves structured Rust error details and recovery action", () => {
    const error = decodeAppError({
      code: "private_network_required",
      message: "Pairing requires a private network.",
      action: "Change the Windows network profile to Private.",
    });

    expect(error).toMatchObject({ code: "private_network_required" });
    expect(error.message).toBe("Pairing requires a private network.");
    expect(error.action).toBe("Change the Windows network profile to Private.");
  });

  it("normalizes Error, string, and malformed rejection values", () => {
    expect(decodeAppError(new Error("offline")).message).toBe("offline");
    expect(decodeAppError("connection refused").message).toBe("connection refused");
    expect(decodeAppError(null).message).toBe("An unexpected application error occurred.");
  });
});
