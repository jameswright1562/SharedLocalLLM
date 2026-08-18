import { describe, expect, it } from "vitest";

import { decodeAppError } from "./errors";

describe("decodeAppError", () => {
  it("preserves structured Rust error details and recovery action", () => {
    const error = decodeAppError({
      code: "manual_peer_endpoint_invalid",
      message: "not-an-address is not a valid Ethernet IP address.",
      action: "Enter the other computer's IPv4 address.",
    });

    expect(error).toMatchObject({ code: "manual_peer_endpoint_invalid" });
    expect(error.message).toBe("not-an-address is not a valid Ethernet IP address.");
    expect(error.action).toBe("Enter the other computer's IPv4 address.");
  });

  it("normalizes Error, string, and malformed rejection values", () => {
    expect(decodeAppError(new Error("offline")).message).toBe("offline");
    expect(decodeAppError("connection refused").message).toBe("connection refused");
    expect(decodeAppError(null).message).toBe("An unexpected application error occurred.");
  });
});
