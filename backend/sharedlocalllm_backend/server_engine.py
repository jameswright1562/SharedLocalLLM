"""Pinned llama-server engine management (dual-engine plan, phase 2).

Owns the child ``llama-server`` process for models that benefit from embedded
MTP speculation, routes chat through its loopback OpenAI-compatible API, and
maps responses onto the app's chat event stream. Loopback-only; the existing
per-install bearer key gates every request.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, BinaryIO, cast

from .errors import BackendError
from .llama_server import install_root_candidates, locate_llama_server, probe_health
from .tool_calls import normalize_tool_message, normalize_tool_stream

START_TIMEOUT_SECONDS = 300
_CONSOLE_LOCK = threading.Lock()
_MODEL_RESPONSE_IDS = itertools.count(1)


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


def log_tail(path: Path, limit: int = 2000, start: int = 0) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            return handle.read().decode("utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""


def server_environment() -> dict[str, str]:
    """Inherit hardware settings, but not configuration for other llama.cpp apps."""
    return {
        name: value for name, value in os.environ.items()
        if not name.upper().startswith("LLAMA_ARG_")
        and name.upper() != "LLAMA_API_KEY"
    }


def _redact_server_output(text: str, model_path: str | None, api_key: str | None) -> str:
    redacted = text.replace(model_path, "<model-path>") if model_path else text
    if api_key:
        redacted = redacted.replace(api_key, "<redacted>")
    roots = {
        str(Path.home()),
        os.environ.get("USERPROFILE", ""),
        os.environ.get("LOCALAPPDATA", ""),
        os.environ.get("APPDATA", ""),
    }
    for root in sorted((value for value in roots if value), key=len, reverse=True):
        redacted = redacted.replace(root, "<private-path>")
    return redacted


def _show_server_console_line(text: str) -> bool:
    return not (
        "create_tensor: loading tensor " in text
        or ("CUDA Graph id " in text and " reused" in text)
    )


def write_model_output(
    content: str,
    *,
    model_path: str | None = None,
    api_key: str | None = None,
    complete: bool = True,
) -> None:
    """Print one redacted answer block without exposing prompts or reasoning."""
    output = _redact_server_output(content, model_path, api_key) or "<no text output>"
    state = "" if complete else " partial"
    with _CONSOLE_LOCK:
        response_id = next(_MODEL_RESPONSE_IDS)
        sys.stderr.write(
            f"\n[model response #{response_id}{state}]\n{output}\n"
            f"[/model response #{response_id}]\n"
        )
        sys.stderr.flush()


def relay_server_output(
    stream: BinaryIO, log_path: Path, model_path: str, api_key: str | None,
) -> None:
    """Keep the full redacted log while hiding repetitive console-only noise."""
    try:
        with log_path.open("ab") as handle:
            while line := stream.readline():
                text = _redact_server_output(
                    line.decode("utf-8", errors="replace"), model_path, api_key
                )
                handle.write(text.encode("utf-8"))
                handle.flush()
                if sys.stderr is not None and _show_server_console_line(text):
                    with _CONSOLE_LOCK:
                        sys.stderr.write(text)
                        sys.stderr.flush()
    except OSError:
        return


def startup_failure_message(detail: str) -> str:
    compact = " ".join(detail.split())
    if "missing result_norm/result_embd tensor" in detail:
        return (
            "llama-server hit the known MTP embedding or reranking mode conflict "
            "(missing result_norm/result_embd tensor). Restart SharedLocalLLM so its "
            "isolated llama-server environment takes effect, then retry."
        )
    return f"llama-server exited during startup. {compact[-1200:]}".strip()


def wire_messages(
    messages: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    allowed = (
        "role", "content", "reasoning_content", "tool_calls",
        "tool_call_id", "function_call", "name",
    )
    wire = [
        {key: item[key] for key in allowed if key in item}
        for item in messages
    ]
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
    gpu_layers: int | None,
    tensor_split: list[int] | None,
    reasoning_preserve: bool,
    load_config: dict[str, Any] | None = None,
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
        "--no-agent",
        "--no-ui",
        "--offline",
    ]
    config = load_config or {}
    command += [
        "--split-mode", "layer",
        "--flash-attn", "on" if config.get("flashAttention") else "off",
        "--batch-size", str(max(1, int(config.get("batchSize", 512)))),
    ]
    if gpu_layers is not None:
        command += ["-ngl", str(max(0, int(gpu_layers))), "--fit", "off"]
        if tensor_split and len(tensor_split) > 1:
            command += [
                "--tensor-split",
                ",".join(str(max(0, int(value))) for value in tensor_split),
            ]
    raw_ubatch = config.get("uBatch")
    if raw_ubatch:
        command += ["--ubatch-size", str(max(1, int(raw_ubatch)))]
    for key, flag in (
        ("kvCacheK", "--cache-type-k"),
        ("kvCacheV", "--cache-type-v"),
    ):
        value = config.get(key)
        if value:
            command += [flag, str(value)]
    # Autotuned RPC-only knobs; absent unless a tuning run stored them.
    if int(config.get("noOpOffload") or 0) == 1:
        command += ["--no-op-offload"]
    raw_poll = config.get("rpcPoll")
    if raw_poll is not None:
        command += ["--poll", str(max(0, min(100, int(raw_poll))))]
    use_mmap = bool(config.get("useMmap", True))
    use_mlock = bool(config.get("useMlock"))
    load_mode = (
        "mmap+mlock" if use_mmap and use_mlock
        else "mmap" if use_mmap
        else "mlock" if use_mlock
        else "none"
    )
    command += ["--load-mode", load_mode]
    cpu_threads = int(config.get("cpuThreads", 0))
    if cpu_threads > 0:
        command += ["--threads", str(cpu_threads)]
    if mtp and speculation_supported:
        command += ["--spec-type", "draft-mtp"]
    if reasoning_preserve:
        command += ["--reasoning-preserve"]
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
        self.model_path: str | None = None
        self.rpc_endpoint: str | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._spec_support: dict[str, bool] = {}
        self._log_thread: threading.Thread | None = None
        self._log_offset = 0

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
                    [key, "--help"], capture_output=True, text=True, timeout=20,
                    env=server_environment(),
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
        gpu_layers: int | None = None,
        tensor_split: list[int] | None = None,
        reasoning_preserve: bool = False,
        load_config: dict[str, Any] | None = None,
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
            gpu_layers=gpu_layers,
            tensor_split=tensor_split,
            reasoning_preserve=reasoning_preserve,
            load_config=load_config,
        )

        log_path = server_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._log_offset = log_path.stat().st_size
        except OSError:
            self._log_offset = 0
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(  # noqa: S603
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags,
            env=server_environment(),
        )
        if self.process.stdout is not None:
            self._log_thread = threading.Thread(
                target=relay_server_output,
                args=(self.process.stdout, log_path, model_path, api_key),
                name="llama-server-log",
                daemon=True,
            )
            self._log_thread.start()
        self.port = port
        self.api_key = api_key
        self.model_id = model_id
        self.model_path = model_path
        self.rpc_endpoint = rpc_endpoint

        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                await self._finish_log_relay()
                detail = log_tail(log_path, start=self._log_offset)
                message = startup_failure_message(detail)
                self.store.log("ERROR", "llama_server_failed", message)
                self.process = None
                self.port = None
                self.model_id = None
                self.model_path = None
                self.rpc_endpoint = None
                raise BackendError(
                    "llama_server_failed",
                    message,
                    "Open the logs folder for the redacted llama-server output.",
                )
            if probe_health(port, api_key):
                self.store.log(
                    "INFO", "llama_server_ready",
                    f"127.0.0.1:{port} ctx={context} mtp={mtp} rpc={rpc_endpoint} "
                    f"gpu_layers={gpu_layers} split={tensor_split} "
                    f"reasoning_preserve={reasoning_preserve}",
                )
                return
            await asyncio.sleep(0.5)
        await self.stop()
        message = "llama-server did not become healthy in time."
        self.store.log("ERROR", "llama_server_timeout", message)
        raise BackendError(
            "llama_server_timeout", message,
            "Open the logs folder for the redacted llama-server output.",
        )

    async def stop(self) -> None:
        self.cancel()
        process, self.process = self.process, None
        self.port = None
        self.model_id = None
        self.model_path = None
        self.rpc_endpoint = None
        if process is None or process.poll() is not None:
            await self._finish_log_relay()
            return
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 10)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 5)
        await self._finish_log_relay()

    async def _finish_log_relay(self) -> None:
        thread, self._log_thread = self._log_thread, None
        if thread is not None and thread is not threading.current_thread():
            await asyncio.to_thread(thread.join, 2)

    def cancel(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _write_model_output(self, content: str, complete: bool = True) -> None:
        write_model_output(
            content,
            model_path=self.model_path,
            api_key=self.api_key,
            complete=complete,
        )

    def _payload(
        self, messages: list[dict[str, Any]], settings: dict[str, Any],
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": wire_messages(messages, settings),
            "temperature": float(settings.get("temperature", 0.7)),
            "max_tokens": int(settings.get("maxTokens", 512)),
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    async def _open_request(
        self, payload: dict[str, Any], timeout: float | None
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
        self, payload: dict[str, Any], timeout: float | None = None
    ) -> tuple[int, str]:
        reader, writer = await self._open_request(payload, timeout)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = await reader.read(64 * 1024) if timeout is None else cast(
                    bytes, await asyncio.wait_for(reader.read(64 * 1024), timeout)
                )
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
        self, messages: list[dict[str, Any]], settings: dict[str, Any],
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        status, body = await self._read_response(
            self._payload(messages, settings, tools, tool_choice)
        )
        if status != 200:
            raise BackendError("generation_failed", _error_message(body, status))
        try:
            data = json.loads(body)
            choice = data["choices"][0]
            message = dict(choice["message"])
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError) as error:
            raise BackendError("generation_failed", f"Unparsable llama-server reply: {error}") from error
        elapsed = max(time.perf_counter() - started, 0.001)
        completion = max(1, int(usage.get("completion_tokens") or 1))
        message, finish_reason = normalize_tool_message(
            message,
            choice.get("finish_reason"),
            tools if tool_choice != "none" else None,
        )
        reasoning = str(message.get("reasoning_content") or "")
        content = str(message.get("content") or "")
        self._write_model_output(content)
        return {
            "content": content,
            "reasoning": reasoning,
            "tokensPerSecond": round(completion / elapsed, 2),
            "message": message,
            "finishReason": finish_reason,
            "usage": usage,
        }

    async def chat_openai_stream(
        self, messages: list[dict[str, Any]], settings: dict[str, Any],
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Relay llama-server's native OpenAI chunks without losing tool calls."""
        payload = self._payload(messages, settings, tools, tool_choice)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        events = self._sse_events(payload)
        output_parts: list[str] = []
        completed = False
        needs_tool_fallback = bool(
            tools and tool_choice != "none" and not isinstance(tool_choice, dict)
        )
        try:
            if needs_tool_fallback and tools:
                buffered = [event async for event in events]
                for event in normalize_tool_stream(buffered, tools):
                    output_parts.append(_openai_event_content(event))
                    yield event
                completed = True
                return
            async for event in events:
                output_parts.append(_openai_event_content(event))
                yield event
            completed = True
        finally:
            if completed or output_parts:
                self._write_model_output("".join(output_parts), completed)

    async def _sse_events(
        self, payload: dict[str, Any], timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        reader, writer = await self._open_request(payload, timeout)
        try:
            status = 0
            buffer = b""
            while True:
                chunk = await reader.read(64 * 1024) if timeout is None else cast(
                    bytes, await asyncio.wait_for(reader.read(64 * 1024), timeout)
                )
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
                        raise BackendError(
                            "generation_failed",
                            _error_message(rest.decode("utf-8", "replace"), status),
                        )
                    buffer = rest
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    event = parse_sse_event(raw_line.decode("utf-8", errors="replace"))
                    if event is not None:
                        yield event
        finally:
            if self._writer is writer:
                self._writer = None
            writer.close()

    async def chat_stream(
        self, messages: list[dict[str, Any]], settings: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._payload(messages, settings)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        started = time.perf_counter()
        first_token_at: float | None = None
        completion_tokens = 0
        finished = False
        output_parts: list[str] = []
        try:
            async for event in self._sse_events(payload):
                choices = event.get("choices") or []
                delta = (choices[0].get("delta") if choices else {}) or {}
                reasoning_piece = delta.get("reasoning_content")
                if reasoning_piece:
                    yield {"type": "reasoning", "content": str(reasoning_piece)}
                content_piece = delta.get("content")
                if content_piece:
                    output_parts.append(str(content_piece))
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
            self.cancel()
            if finished or output_parts:
                self._write_model_output("".join(output_parts), finished)

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


def _openai_event_content(event: dict[str, Any]) -> str:
    choices = event.get("choices") or []
    delta = (choices[0].get("delta") if choices else {}) or {}
    return str(delta.get("content") or "")
