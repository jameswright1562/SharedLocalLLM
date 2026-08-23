from __future__ import annotations

import json

from sharedlocalllm_backend.tool_calls import parse_text_tool_calls


TOOLS = [{
    "type": "function",
    "function": {"name": "Bash", "parameters": {"type": "object"}},
}]


def test_parses_qwen_xml_parameter_tool_call() -> None:
    text = """<tool_call>
<function=Bash>
<parameter=command>
$c = Get-Content -LiteralPath \"C:\\Temp\\CustomCurve.il\" -ReadCount 0
</parameter>
</function>
</tool_call>"""

    parsed = parse_text_tool_calls(text, TOOLS)
    assert parsed is not None
    content, calls = parsed
    assert content is None
    assert calls[0]["function"]["name"] == "Bash"
    assert json.loads(calls[0]["function"]["arguments"])["command"].startswith("$c = Get-Content")


def test_parses_standard_qwen_json_tool_call() -> None:
    text = '<tool_call>{"name":"Bash","arguments":{"command":"Get-Location"}}</tool_call>'
    content, calls = parse_text_tool_calls(text, TOOLS) or ("missing", [])
    assert content is None
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "Get-Location"}


def test_preserves_text_around_tool_calls() -> None:
    text = 'Checking. <tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>'
    content, calls = parse_text_tool_calls(text, TOOLS) or (None, [])
    assert content == "Checking."
    assert len(calls) == 1


def test_does_not_parse_unknown_or_malformed_tools() -> None:
    unknown = '<tool_call>{"name":"DeleteEverything","arguments":{}}</tool_call>'
    malformed = "<tool_call><function=Bash>missing parameters</function></tool_call>"
    assert parse_text_tool_calls(unknown, TOOLS) is None
    assert parse_text_tool_calls(malformed, TOOLS) is None
