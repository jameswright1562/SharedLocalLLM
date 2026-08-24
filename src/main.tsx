import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { MantineProvider } from "@mantine/core";
import { ContextMenuProvider } from "mantine-contextmenu";
import "@mantine/core/styles.css";
import "mantine-contextmenu/styles.css";
import "./styles/global.css";
import { theme } from "./theme";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="dark" forceColorScheme="dark">
      <ContextMenuProvider>
        <App />
      </ContextMenuProvider>
    </MantineProvider>
  </React.StrictMode>,
);
