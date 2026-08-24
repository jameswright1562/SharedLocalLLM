from __future__ import annotations

from typing import Any, Iterable, Iterator

from .errors import BackendError
from .reasoning import ReasoningStreamSplitter


def request_tool_options(
    body: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, Any]:
    """Validate the OpenAI tool fields before llama-cpp-python sees them."""
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise BackendError("api_tools_invalid", "tools must be an array of tool definitions.")
        for tool in tools:
            function = tool.get("function")
            if tool.get("type") != "function" or not isinstance(function, dict) or not function.get("name"):
                raise BackendError(
                    "api_tools_invalid", "Every tool must contain a named function definition."
                )

    tool_choice = body.get("tool_choice")
    if tool_choice is None:
        return tools, None
    if isinstance(tool_choice, str):
        if tool_choice not in ("auto", "none", "required"):
            raise BackendError("api_tool_choice_invalid", "tool_choice is not supported.")
        if tool_choice == "required" and not tools:
            raise BackendError("api_tool_choice_invalid", "tool_choice 'required' needs at least one tool.")
        return tools, tool_choice
    if not isinstance(tool_choice, dict):
        raise BackendError("api_tool_choice_invalid", "tool_choice must name a function.")
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    available = {
        tool["function"]["name"] for tool in tools or []
        if isinstance(tool.get("function"), dict)
    }
    if tool_choice.get("type") != "function" or not name or name not in available:
        raise BackendError(
            "api_tool_choice_invalid", "tool_choice must name one of the supplied functions."
        )
    return tools, tool_choice


def completion_message(response: dict[str, Any]) -> dict[str, Any]:
    native = response.get("message")
    message = dict(native) if isinstance(native, dict) else {
        "role": "assistant", "content": response.get("content", "")
    }
    if response.get("reasoning") and "reasoning_content" not in message:
        message["reasoning_content"] = response["reasoning"]
    return message


def completion_finish_reason(response: dict[str, Any]) -> str:
    return str(response.get("finishReason") or "stop")


def completion_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if isinstance(usage, dict):
        return usage
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def chunk_payload(
    choices: list[dict[str, Any]], model: str,
    usage: dict[str, Any] | None = None, include_usage: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-sharedlocalllm",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": choices,
    }
    if include_usage:
        payload["usage"] = usage
    return payload


def reasoning_stream_chunks(
    chunks: Iterable[dict[str, Any]], reasoning: bool,
) -> Iterator[dict[str, Any]]:
    """Map reasoning text to reasoning_content without leaking its markers."""
    if not reasoning:
        yield from chunks
        return
    splitter = ReasoningStreamSplitter(True)
    terminal: list[dict[str, Any]] = []

    def with_delta(
        chunk: dict[str, Any], choice: dict[str, Any], delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            **chunk,
            "choices": [{**choice, "delta": delta, "finish_reason": finish_reason}],
        }

    for chunk in chunks:
        choices = chunk.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0], dict):
            yield chunk
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        text = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(text, str):
            base = {key: value for key, value in delta.items() if key != "content"}
            if base:
                yield with_delta(chunk, choice, base)
            for kind, piece in splitter.push(text):
                field = "reasoning_content" if kind == "reasoning" else "content"
                if piece:
                    yield with_delta(chunk, choice, {field: piece})
        elif choice.get("finish_reason") is not None:
            terminal.append(chunk)
        else:
            yield chunk
    for kind, piece in splitter.finish():
        field = "reasoning_content" if kind == "reasoning" else "content"
        if piece:
            yield {
                "choices": [{
                    "index": 0, "delta": {field: piece}, "finish_reason": None,
                }],
            }
    yield from terminal
