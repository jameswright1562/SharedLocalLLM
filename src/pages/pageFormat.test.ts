import { describe, expect, it } from "vitest";

import { formatRunTime } from "./pageFormat";

describe("formatRunTime", () => {
  it("formats ISO timestamps", () => {
    expect(formatRunTime("2026-08-13T10:00:00.000Z")).not.toBe("—");
  });

  it("interprets epoch-seconds values produced by the backend", () => {
    const seconds = Math.floor(Date.now() / 1000);
    expect(formatRunTime(String(seconds))).not.toBe("—");
  });

  it("falls back for empty, missing, and invalid values", () => {
    expect(formatRunTime("")).toBe("—");
    expect(formatRunTime(undefined)).toBe("—");
    expect(formatRunTime(null)).toBe("—");
    expect(formatRunTime("not-a-date")).toBe("—");
  });
});
