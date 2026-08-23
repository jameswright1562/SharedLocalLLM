from __future__ import annotations

from typing import Any, Iterator

from .errors import BackendError


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


def chunk_payload(choices: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-sharedlocalllm",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": choices,
    }


def buffered_stream_choices(response: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
    """Turn a peer's buffered completion into valid OpenAI stream choices."""
    message = completion_message(response)
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message.get("content", "")
    if message.get("reasoning_content"):
        delta["reasoning_content"] = message["reasoning_content"]
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        delta["tool_calls"] = [
            {**call, "index": index} for index, call in enumerate(tool_calls)
            if isinstance(call, dict)
        ]
    yield [{"index": 0, "delta": delta, "finish_reason": None}]
    yield [{
        "index": 0, "delta": {},
        "finish_reason": completion_finish_reason(response),
    }]
