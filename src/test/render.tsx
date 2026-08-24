import { render as rtlRender } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";
import { ContextMenuProvider } from "mantine-contextmenu";
import type { ReactElement } from "react";

import { theme } from "../theme";

function wrap(element: ReactElement) {
  return (
    <MantineProvider theme={theme} forceColorScheme="dark">
      <ContextMenuProvider>{element}</ContextMenuProvider>
    </MantineProvider>
  );
}

export function render(element: ReactElement) {
  const utils = rtlRender(wrap(element));
  return {
    ...utils,
    rerender: (next: ReactElement) => utils.rerender(wrap(next)),
  };
}
