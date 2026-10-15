import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    host: host ?? "127.0.0.1",
    port: 1420,
    strictPort: true,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // IDE and tooling dirs (Cursor/AionUi skills use junctions that can dangle
      // and crash the watcher with UNKNOWN stat errors); never source files.
      // The dot-dir pattern covers .cursor, .aionrs, .idea, .vscode and similar.
      ignored: ["**/src-tauri/**", "**/node_modules/**", "**/.git/**", "**/.*/**"],
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
});
