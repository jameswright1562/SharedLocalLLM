// Cross-platform entry point for the `bun backend:*` scripts.
//
// Windows runs the existing PowerShell scripts; macOS/Linux run the
// POSIX shell equivalents. Behaviour on Windows is unchanged.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_SCRIPTS: Record<string, string> = {
  install: "setup-python-backend",
  ensure: "ensure-python-backend",
  package: "build-python-backend",
  "install-llama-server": "install-llama-server",
};

function venvPython(root: string): string {
  return process.platform === "win32"
    ? path.join(root, "backend", ".venv", "Scripts", "python.exe")
    : path.join(root, "backend", ".venv", "bin", "python");
}

function run(command: string, args: string[]): number {
  const result = spawnSync(command, args, { stdio: "inherit" });
  if (result.error) {
    console.error(String(result.error));
    return 1;
  }
  return result.status ?? 1;
}

const name = process.argv[2];
const extra = process.argv.slice(3);
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

if (name === "typecheck") {
  const python = venvPython(root);
  if (!existsSync(python)) {
    console.error("The backend virtualenv is missing. Run `bun backend:install` first.");
    process.exit(1);
  }
  process.exit(run(python, ["-m", "pyright", ...extra]));
}

if (name === "test") {
  const python = venvPython(root);
  if (!existsSync(python)) {
    console.error("The backend virtualenv is missing. Run `bun backend:install` first.");
    process.exit(1);
  }
  process.exit(
    run(python, [
      "-m",
      "pytest",
      "backend/tests",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "backend/.pytest-tmp",
      ...extra,
    ]),
  );
}

const base = SCRIPT_SCRIPTS[name];
if (!base) {
  console.error(`Unknown backend script: ${name ?? "(none)"}`);
  process.exit(2);
}

if (process.platform === "win32") {
  process.exit(
    run("powershell", [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      path.join(root, "scripts", `${base}.ps1`),
      ...extra,
    ]),
  );
}

const script = path.join(root, "scripts", `${base}.sh`);
if (!existsSync(script)) {
  console.error(`Missing backend script for this platform: ${script}`);
  process.exit(1);
}
process.exit(run("sh", [script, ...extra]));
