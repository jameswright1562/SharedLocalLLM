from __future__ import annotations

import json
import re
import secrets
from typing import Any

_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_BLOCK = re.compile(
    r"\A\s*<function=([^>\r\n]+)>\s*(.*?)\s*</function>\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_PARAMETER = re.compile(
    r"<parameter=([^>\r\n]+)>\s*(.*?)\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(tool["function"]["name"])
        for tool in tools
        if isinstance(tool.get("function"), dict) and tool["function"].get("name")
    }


def _tool_call(name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{secrets.token_hex(12)}",
        "type": "function",
        "function": {"name": name, "arguments": encoded},
    }


def _parse_block(body: str, available: set[str]) -> dict[str, Any] | None:
    function_match = _FUNCTION_BLOCK.match(body)
    if function_match:
        name = function_match.group(1).strip()
        parameters = {
            match.group(1).strip(): match.group(2).strip("\r\n")
            for match in _PARAMETER.finditer(function_match.group(2))
        }
        if name in available and parameters:
            return _tool_call(name, parameters)
        return None

    payload = body.strip()
    if payload.startswith("```json") and payload.endswith("```"):
        payload = payload[7:-3].strip()
    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments", {})
    if name not in available or not isinstance(arguments, (dict, str)):
        return None
    return _tool_call(str(name), arguments)


def parse_text_tool_calls(
    text: str, tools: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]] | None:
    """Parse known Qwen tool markup, limited to functions the client supplied."""
    available = _tool_names(tools)
    calls: list[dict[str, Any]] = []
    remaining: list[str] = []
    cursor = 0
    for match in _TOOL_BLOCK.finditer(text):
        call = _parse_block(match.group(1), available)
        if call is None:
            continue
        remaining.append(text[cursor:match.start()])
        cursor = match.end()
        calls.append(call)
    if not calls:
        return None
    remaining.append(text[cursor:])
    content = "".join(remaining).strip()
    return content or None, calls


def normalize_tool_message(
    message: dict[str, Any], finish_reason: str | None,
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], str]:
    if message.get("tool_calls") or not tools or not isinstance(message.get("content"), str):
        return message, finish_reason or "stop"
    parsed = parse_text_tool_calls(message["content"], tools)
    if parsed is None:
        return message, finish_reason or "stop"
    content, calls = parsed
    return {**message, "content": content, "tool_calls": calls}, "tool_calls"


def normalize_tool_stream(
    chunks: list[dict[str, Any]], tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert buffered text markup to OpenAI tool deltas when native parsing did not."""
    choices = [
        choice for chunk in chunks for choice in chunk.get("choices", [])
        if isinstance(choice, dict)
    ]
    if any(choice.get("delta", {}).get("tool_calls") for choice in choices):
        return chunks
    text = "".join(
        str(choice.get("delta", {}).get("content") or "") for choice in choices
    )
    parsed = parse_text_tool_calls(text, tools)
    if parsed is None:
        return chunks
    content, calls = parsed
    normalized = [{
        "choices": [{
            "index": 0, "delta": {"role": "assistant"}, "finish_reason": None,
        }],
    }]
    if content:
        normalized.append({
            "choices": [{
                "index": 0, "delta": {"content": content}, "finish_reason": None,
            }],
        })
    normalized.append({
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{**call, "index": index} for index, call in enumerate(calls)]},
            "finish_reason": None,
        }],
    })
    normalized.append({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    })
    return normalized
