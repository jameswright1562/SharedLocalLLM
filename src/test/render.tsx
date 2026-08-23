import { render as rtlRender } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";
import type { ReactElement } from "react";

import { theme } from "../theme";

export function render(element: ReactElement) {
  return rtlRender(
    <MantineProvider theme={theme} forceColorScheme="dark">
      {element}
    </MantineProvider>,
  );
}
