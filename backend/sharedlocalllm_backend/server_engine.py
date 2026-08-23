"""Pinned llama-server engine management (dual-engine plan, phase 2).

Owns the child ``llama-server`` process for models that benefit from embedded
MTP speculation, routes chat through its loopback OpenAI-compatible API, and
maps responses onto the app's chat event stream. Loopback-only; the existing
per-install bearer key gates every request.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator

from .errors import BackendError
from .llama_server import install_root_candidates, locate_llama_server, probe_health

START_TIMEOUT_SECONDS = 300


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_log_path() -> Path:
    base = (
        os.environ.get("LOCALAPPDATA")
        if os.name == "nt"
        else os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    )
    return Path(base or Path.home()) / "SharedLocalLLM" / "logs" / "llama-server.log"


def log_tail(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""


def wire_messages(
    messages: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, str]]:
    wire = [{"role": item.get("role"), "content": item.get("content", "")} for item in messages]
    system = str(settings.get("systemPrompt", "")).strip()
    if system and not any(item["role"] == "system" for item in wire):
        wire.insert(0, {"role": "system", "content": system})
    return wire


def parse_sse_event(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
        return None
    try:
        value = json.loads(line[5:])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def build_command(
    exe: Path,
    *,
    model_path: str,
    port: int,
    context: int,
    api_key: str | None,
    mtp: bool,
    speculation_supported: bool,
    rpc_endpoint: str | None,
) -> list[str]:
    """Assemble the pinned llama-server invocation for this launch."""
    command = [
        str(exe),
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(context),
        "--parallel", "1",
        "--jinja",
    ]
    # Full offload: placement across local GPU + RPC devices happens inside
    # llama.cpp (layer split); RPC endpoints arrive via --rpc.
    command += ["-ngl", "99"]
    if mtp and speculation_supported:
        command += ["--spec-type", "draft-mtp"]
    if rpc_endpoint:
        command += ["--rpc", rpc_endpoint]
    if api_key:
        command += ["--api-key", api_key]
    return command


class ServerEngine:
    """Lifecycle + chat client for the pinned llama-server child process."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.process: subprocess.Popen[bytes] | None = None
        self.port: int | None = None
        self.api_key: str | None = None
        self.model_id: str | None = None
        self.rpc_endpoint: str | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._spec_support: dict[str, bool] = {}

    @property
    def active(self) -> bool:
        return (
            self.process is not None
            and self.process.poll() is None
            and self.port is not None
        )

    def available(self) -> Path | None:
        return locate_llama_server(install_root_candidates())

    def supports_speculation(self, exe: Path) -> bool:
        key = str(exe)
        if key not in self._spec_support:
            try:
                result = subprocess.run(
                    [key, "--help"], capture_output=True, text=True, timeout=20
                )
                combined = f"{result.stdout}{result.stderr}"
                self._spec_support[key] = "--spec-type" in combined
            except (OSError, subprocess.TimeoutExpired):
                self._spec_support[key] = False
        return self._spec_support[key]

    async def start(
        self,
        *,
        exe: Path,
        model_path: str,
        model_id: str,
        context: int,
        api_key: str | None,
        mtp: bool,
        rpc_endpoint: str | None = None,
    ) -> None:
        await self.stop()
        port = free_port()
        command = build_command(
            exe,
            model_path=model_path,
            port=port,
            context=context,
            api_key=api_key,
            mtp=mtp,
            speculation_supported=self.supports_speculation(exe),
            rpc_endpoint=rpc_endpoint,
        )

        log_path = server_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as handle:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(  # noqa: S603
                command, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, creationflags=flags,
            )
        self.port = port
        self.api_key = api_key
        self.model_id = model_id
        self.rpc_endpoint = rpc_endpoint

        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                detail = log_tail(log_path)
                self.process = None
                self.port = None
                self.rpc_endpoint = None
                raise BackendError(
                    "llama_server_failed",
                    f"llama-server exited during startup. {detail}",
                )
            if probe_health(port, api_key):
                self.store.log(
                    "INFO", "llama_server_ready",
                    f"127.0.0.1:{port} ctx={context} mtp={mtp} rpc={rpc_endpoint}",
                )
                return
            await asyncio.sleep(0.5)
        await self.stop()
        raise BackendError(
            "llama_server_timeout", "llama-server did not become healthy in time."
        )

    async def stop(self) -> None:
        self._writer = None
        process, self.process = self.process, None
        self.port = None
        self.model_id = None
        self.rpc_endpoint = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 10)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 5)

    def cancel(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _payload(self, messages: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": wire_messages(messages, settings),
            "temperature": float(settings.get("temperature", 0.7)),
            "max_tokens": int(settings.get("maxTokens", 512)),
        }

    async def _open_request(
        self, payload: dict[str, Any], timeout: float
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if not self.port:
            raise BackendError("model_not_loaded", "The llama-server engine is not running.")
        body = json.dumps(payload).encode()
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            **self._auth_header(),
        }
        head = "POST /v1/chat/completions HTTP/1.1\r\nConnection: close\r\n" + "".join(
            f"{name}: {value}\r\n" for name, value in headers.items()
        )
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.port, limit=8 * 1024 * 1024), 10
            )
            writer.write(head.encode() + b"\r\n" + body)
            await writer.drain()
        except (OSError, asyncio.TimeoutError) as error:
            raise BackendError("llama_server_unreachable", str(error)) from error
        self._writer = writer
        return reader, writer

    async def _read_response(
        self, payload: dict[str, Any], timeout: float = 600
    ) -> tuple[int, str]:
        reader, writer = await self._open_request(payload, timeout)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = await asyncio.wait_for(reader.read(64 * 1024), timeout)
                if not chunk:
                    break
                chunks.append(chunk)
            text = b"".join(chunks).decode("utf-8", errors="replace")
            header_block, _, body = text.partition("\r\n\r\n")
            parts = header_block.splitlines()[0].split() if header_block else []
            status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return status, body
        finally:
            self._writer = None
            writer.close()

    async def chat(
        self, messages: list[dict[str, Any]], settings: dict[str, Any]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        status, body = await self._read_response(self._payload(messages, settings))
        if status != 200:
            raise BackendError("generation_failed", _error_message(body, status))
        try:
            data = json.loads(body)
            message = data["choices"][0]["message"]
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError) as error:
            raise BackendError("generation_failed", f"Unparsable llama-server reply: {error}") from error
        elapsed = max(time.perf_counter() - started, 0.001)
        completion = max(1, int(usage.get("completion_tokens") or 1))
        return {
            "content": str(message.get("content") or ""),
            "reasoning": str(message.get("reasoning_content") or ""),
            "tokensPerSecond": round(completion / elapsed, 2),
        }

    async def chat_stream(
        self, messages: list[dict[str, Any]], settings: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._payload(messages, settings)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        reader, writer = await self._open_request(payload, timeout=600)
        started = time.perf_counter()
        first_token_at: float | None = None
        completion_tokens = 0
        buffer = b""
        finished = False
        try:
            status = 0
            while True:
                chunk = await reader.read(64 * 1024)
                if not chunk:
                    break
                buffer += chunk
                if status == 0:
                    head, separator, rest = buffer.partition(b"\r\n\r\n")
                    if not separator:
                        continue
                    parts = head.decode("latin-1").splitlines()[0].split()
                    status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    if status != 200:
                        raise BackendError("generation_failed", _error_message(rest.decode("utf-8", "replace"), status))
                    buffer = rest
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    event = parse_sse_event(raw_line.decode("utf-8", errors="replace"))
                    if not event:
                        continue
                    choices = event.get("choices") or []
                    delta = (choices[0].get("delta") if choices else {}) or {}
                    reasoning_piece = delta.get("reasoning_content")
                    if reasoning_piece:
                        yield {"type": "reasoning", "content": str(reasoning_piece)}
                    content_piece = delta.get("content")
                    if content_piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        yield {"type": "token", "content": str(content_piece)}
                    usage = event.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens"):
                        completion_tokens = int(usage["completion_tokens"])
            if not finished:
                window = max(time.perf_counter() - (first_token_at or started), 0.001)
                if completion_tokens:
                    yield {"type": "stats", "tokensPerSecond": round(completion_tokens / window, 2)}
                yield {"type": "done"}
                finished = True
        finally:
            self._writer = None
            writer.close()

    async def benchmark(self) -> tuple[float, float]:
        result = await self.chat(
            [{"role": "user", "content": "SharedLocalLLM benchmark: explain local inference."}],
            {"temperature": 0.0, "maxTokens": 128},
        )
        return 0.0, float(result.get("tokensPerSecond") or 0.0)


def _error_message(body: str, status: int) -> str:
    try:
        parsed = json.loads(body)
        return str(parsed.get("error", {}).get("message") or body[:400])
    except ValueError:
        return f"llama-server returned HTTP {status}."
