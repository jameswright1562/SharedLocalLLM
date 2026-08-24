from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from .errors import BackendError


async def request_stream(
    host: str,
    port: int,
    version: int,
    op: str,
    data: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Read an unbounded sequence of peer events from one connection."""
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, limit=2 * 1024 * 1024), 5
        )
        await _write_frame(writer, {"version": version, "op": op, "data": data})
        while True:
            frame = await _read_frame(reader)
            if frame.get("ok") is not True:
                code = frame.get("code")
                message = frame.get("message")
                action = frame.get("action")
                raise BackendError(
                    code if isinstance(code, str) else "peer_error",
                    message if isinstance(message, str) else "Peer stream failed",
                    action if isinstance(action, str) else None,
                )
            if frame.get("done") is True:
                return
            event = frame.get("event")
            if not isinstance(event, dict):
                raise BackendError("peer_protocol", "The peer returned an invalid stream event.")
            yield event
    except BackendError:
        raise
    except (OSError, asyncio.TimeoutError, ValueError) as error:
        raise BackendError(
            "peer_unavailable", f"The other computer did not answer: {error}"
        ) from error
    finally:
        if writer:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


async def serve_events(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    events: AsyncIterator[dict[str, Any]],
) -> None:
    """Relay events while also noticing a consumer that closes its socket."""
    disconnected = asyncio.create_task(reader.read())
    iterator = events.__aiter__()
    next_event: asyncio.Future[dict[str, Any]] | None = None
    try:
        while True:
            next_event = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait(
                (disconnected, next_event), return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in done:
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
                return
            try:
                event = next_event.result()
            except StopAsyncIteration:
                await _write_frame(writer, {"ok": True, "done": True})
                return
            next_event = None
            if not isinstance(event, dict):
                raise BackendError("peer_protocol", "A peer stream event must be an object.")
            try:
                await _write_frame(writer, {"ok": True, "event": event})
            except OSError:
                return
    finally:
        if next_event is not None:
            if not next_event.done():
                next_event.cancel()
            await asyncio.gather(next_event, return_exceptions=True)
        disconnected.cancel()
        await asyncio.gather(disconnected, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    try:
        line = await reader.readline()
    except ValueError as error:
        raise BackendError("peer_protocol", "The peer returned an oversized frame.") from error
    if not line:
        raise BackendError("peer_closed", "Peer closed the connection.")
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BackendError("peer_protocol", "The peer returned invalid JSON.") from error
    if not isinstance(value, dict):
        raise BackendError("peer_protocol", "The peer response must be a JSON object.")
    return value


async def _write_frame(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
    writer.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")
    await writer.drain()
