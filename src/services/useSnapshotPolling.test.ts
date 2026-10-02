import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useSnapshotPolling } from "./useSnapshotPolling";

afterEach(() => vi.useRealTimers());

describe("snapshot polling", () => {
  it("keeps retrying loading refreshes after a failure", async () => {
    vi.useFakeTimers();
    const refresh = vi
      .fn()
      .mockRejectedValueOnce(new Error("temporary failure"))
      .mockResolvedValue(undefined);
    const { unmount } = renderHook(() => useSnapshotPolling(refresh, true));
    await act(() => vi.advanceTimersByTimeAsync(4500));
    expect(refresh).toHaveBeenCalledTimes(3);
    unmount();
    await act(() => vi.advanceTimersByTimeAsync(8000));
    expect(refresh).toHaveBeenCalledTimes(3);
  });

  it("adapts its interval to loading without overlapping slow requests", async () => {
    vi.useFakeTimers();
    let complete: (() => void) | undefined;
    const refresh = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            complete = resolve;
          }),
      )
      .mockResolvedValue(undefined);
    const { rerender, unmount } = renderHook(
      ({ loading }) => useSnapshotPolling(refresh, loading),
      { initialProps: { loading: false } },
    );
    await act(() => vi.advanceTimersByTimeAsync(7999));
    expect(refresh).not.toHaveBeenCalled();
    await act(() => vi.advanceTimersByTimeAsync(1));
    await act(() => vi.advanceTimersByTimeAsync(16000));
    expect(refresh).toHaveBeenCalledTimes(1);
    await act(async () => {
      complete?.();
    });
    rerender({ loading: true });
    await act(() => vi.advanceTimersByTimeAsync(1500));
    expect(refresh).toHaveBeenCalledTimes(2);
    unmount();
  });
});
