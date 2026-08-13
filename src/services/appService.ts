import type { AppService } from "../types";
import { demoService } from "./demoService";
import { nativeService } from "./nativeService";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

export { demoService } from "./demoService";
export { nativeService } from "./nativeService";

export const appService: AppService =
  typeof window !== "undefined" && window.__TAURI_INTERNALS__ ? nativeService : demoService;
