from __future__ import annotations

import copy
import json
import re
import secrets
from collections.abc import Iterable, Iterator
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
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def _decoded_mapping(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    return decoded if isinstance(decoded, dict) else value


def template_tool_inputs(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Adapt OpenAI JSON strings for GGUF templates that require mappings."""
    normalized_messages = copy.deepcopy(messages)
    for message in normalized_messages:
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and "arguments" in function:
                function["arguments"] = _decoded_mapping(function["arguments"])

    normalized_tools = copy.deepcopy(tools)
    for tool in normalized_tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and "parameters" in function:
            function["parameters"] = _decoded_mapping(function["parameters"])
    return normalized_messages, normalized_tools


def _tool_definitions(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(tool["function"]["name"]): tool["function"]
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


def _parse_block(
    body: str, available: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    function_match = _FUNCTION_BLOCK.match(body)
    if function_match:
        name = function_match.group(1).strip()
        parameter_body = function_match.group(2)
        matches = list(_PARAMETER.finditer(parameter_body))
        if _PARAMETER.sub("", parameter_body).strip():
            return None
        schema = available.get(name, {}).get("parameters", {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        parameters: dict[str, Any] = {}
        for match in matches:
            key = match.group(1).strip()
            value: Any = match.group(2).strip("\r\n")
            property_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
            declared_type = property_schema.get("type") if isinstance(property_schema, dict) else None
            if declared_type and declared_type != "string":
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            parameters[key] = value
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
    available = _tool_definitions(tools)
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


class ToolStreamNormalizer:
    """Incrementally convert textual tool markup while native deltas pass through."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.tools = tools
        self.pending: list[dict[str, Any]] = []
        self.pending_text = ""
        self.native = False
        self.converted = False
        self.role_seen = False
        self.call_index = 0

    def push(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        choices = chunk.get("choices", [])
        native = any(
            isinstance(choice, dict)
            and isinstance(choice.get("delta"), dict)
            and choice["delta"].get("tool_calls")
            for choice in choices
        )
        if native:
            output = self._flush_pending()
            self.native = True
            output.append(chunk)
            self._note_role(chunk)
            return output
        if self.native:
            self._note_role(chunk)
            return [chunk]

        choice = choices[0] if len(choices) == 1 else None
        delta = choice.get("delta") if isinstance(choice, dict) else None
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str):
            assert isinstance(choice, dict) and isinstance(delta, dict)
            self.pending.append(chunk)
            self.pending_text += content
            has_tool, complete = self._text_state(self.pending_text.lower())
            if not complete:
                return []
            output = self._convert_pending() if has_tool else self._flush_pending()
            return [self._tool_finish(item) for item in output] if self.converted else output

        terminal = any(
            isinstance(item, dict) and item.get("finish_reason") is not None
            for item in choices
        ) or "usage" in chunk
        output = self._flush_pending() if terminal else []
        if self.converted:
            chunk = self._tool_finish(chunk)
        output.append(chunk)
        self._note_role(chunk)
        return output

    def finish(self) -> list[dict[str, Any]]:
        output = self._flush_pending()
        return [self._tool_finish(item) for item in output] if self.converted else output

    @staticmethod
    def _text_state(text: str) -> tuple[bool, bool]:
        position = 0
        has_tool = False
        while (start := text.find(_TOOL_OPEN, position)) >= 0:
            has_tool = True
            end = text.find(_TOOL_CLOSE, start + len(_TOOL_OPEN))
            if end < 0:
                return has_tool, False
            position = end + len(_TOOL_CLOSE)
        partial = any(text.endswith(_TOOL_OPEN[:n]) for n in range(1, len(_TOOL_OPEN)))
        return has_tool, not partial

    def _flush_pending(self) -> list[dict[str, Any]]:
        output, self.pending = self.pending, []
        self.pending_text = ""
        for chunk in output:
            self._note_role(chunk)
        return output

    @staticmethod
    def _tool_finish(chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            **chunk,
            "choices": [
                {**choice, "finish_reason": "tool_calls"}
                if isinstance(choice, dict) and choice.get("finish_reason") is not None
                else choice for choice in chunk.get("choices", [])
            ],
        }

    def _convert_pending(self) -> list[dict[str, Any]]:
        parsed = parse_text_tool_calls(self.pending_text, self.tools)
        if parsed is None:
            return self._flush_pending()
        content, calls = parsed
        pending, self.pending = self.pending, []
        self.pending_text = ""
        output: list[dict[str, Any]] = []
        terminal: list[dict[str, Any]] = []
        usage: list[dict[str, Any]] = []
        for chunk in pending:
            choice = chunk["choices"][0]
            delta = choice["delta"]
            siblings = {key: value for key, value in delta.items() if key != "content"}
            envelope = {key: value for key, value in chunk.items() if key != "usage"}
            if siblings or set(choice) - {"index", "delta", "finish_reason"}:
                auxiliary = {**envelope, "choices": [
                    {**choice, "delta": siblings, "finish_reason": None},
                ]}
                output.append(auxiliary)
                self._note_role(auxiliary)
            if choice.get("finish_reason") is not None:
                terminal.append({
                    **envelope,
                    "choices": [{"index": choice.get("index", 0), "delta": {},
                                 "finish_reason": "tool_calls"}],
                })
            if "usage" in chunk:
                usage.append({**envelope, "choices": [], "usage": chunk["usage"]})
        if not self.role_seen:
            output.append({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            self.role_seen = True
        if content:
            output.append({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        indexed_calls = [
            {**call, "index": self.call_index + index} for index, call in enumerate(calls)
        ]
        self.call_index += len(calls)
        output.append({"choices": [
            {"index": 0, "delta": {"tool_calls": indexed_calls}, "finish_reason": None},
        ]})
        self.converted = True
        return [*output, *terminal, *usage]

    def _note_role(self, chunk: dict[str, Any]) -> None:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict) and delta.get("role") == "assistant":
                self.role_seen = True


def normalize_tool_stream(
    chunks: Iterable[dict[str, Any]], tools: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    normalizer = ToolStreamNormalizer(tools)
    for chunk in chunks:
        yield from normalizer.push(chunk)
    yield from normalizer.finish()
