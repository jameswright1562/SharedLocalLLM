import { useEffect } from "react";

/** One app-wide polling loop; failed refreshes retain the retry schedule. */
export function useSnapshotPolling(refreshSnapshot: () => Promise<void>, loading: boolean) {
  useEffect(() => {
    let active = true;
    let timer = 0;
    const schedule = () => {
      timer = window.setTimeout(
        () => {
          void refreshSnapshot()
            // The app's refresh callback reports errors to its status banner.
            .catch(() => undefined)
            .finally(() => {
              if (active) schedule();
            });
        },
        loading ? 1500 : 8000,
      );
    };
    schedule();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [refreshSnapshot, loading]);
}
