from __future__ import annotations

import json

from sharedlocalllm_backend.tool_calls import (
    normalize_tool_stream,
    parse_text_tool_calls,
    template_tool_inputs,
)


TOOLS = [{
    "type": "function",
    "function": {"name": "Bash", "parameters": {"type": "object"}},
}]


def test_parses_qwen_xml_parameter_tool_call() -> None:
    text = """<tool_call>
<function=Bash>
<parameter=command>
$c = Get-Content -LiteralPath "C:\\Temp\\CustomCurve.il" -ReadCount 0
</parameter>
</function>
</tool_call>"""

    parsed = parse_text_tool_calls(text, TOOLS)
    assert parsed is not None
    content, calls = parsed
    assert content is None
    assert calls[0]["function"]["name"] == "Bash"
    assert json.loads(calls[0]["function"]["arguments"])["command"].startswith(
        "$c = Get-Content"
    )


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


def test_xml_parameters_follow_declared_json_schema_types() -> None:
    tools = [{
        "type": "function",
        "function": {
            "name": "SetOptions",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"},
                },
            },
        },
    }]
    text = (
        "<tool_call><function=SetOptions>"
        "<parameter=enabled>true</parameter>"
        "<parameter=count>3</parameter>"
        "</function></tool_call>"
    )
    _, calls = parse_text_tool_calls(text, tools) or (None, [])
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "enabled": True,
        "count": 3,
    }


def test_template_inputs_decode_openai_argument_strings_without_mutating_history() -> None:
    messages = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
        }],
    }]
    tools = [{
        "type": "function",
        "function": {"name": "Bash", "parameters": '{"type":"object"}'},
    }]

    normalized_messages, normalized_tools = template_tool_inputs(messages, tools)
    assert normalized_tools is not None
    assert normalized_messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "command": "pwd"
    }
    assert normalized_tools[0]["function"]["parameters"] == {"type": "object"}
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)
    assert isinstance(tools[0]["function"]["parameters"], str)


def content_chunk(content: str) -> dict:
    return {
        "choices": [{
            "index": 0, "delta": {"content": content}, "finish_reason": None,
        }],
    }


def test_ordinary_text_yields_before_the_source_completes() -> None:
    source_advanced = False

    def chunks():
        nonlocal source_advanced
        yield content_chunk("ordinary answer")
        source_advanced = True
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    normalized = normalize_tool_stream(chunks(), TOOLS)
    assert next(normalized) == content_chunk("ordinary answer")
    assert source_advanced is False


def test_split_text_markup_becomes_tool_deltas_and_tool_finish() -> None:
    chunks = [
        content_chunk('<tool_call>{"name":"Bash",'),
        content_chunk('"arguments":{"command":"pwd"}}</tool_call>'),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]

    normalized = list(normalize_tool_stream(chunks, TOOLS))

    calls = [
        choice["delta"]["tool_calls"]
        for chunk in normalized
        for choice in chunk.get("choices", [])
        if choice.get("delta", {}).get("tool_calls")
    ]
    assert calls[0][0]["function"]["name"] == "Bash"
    assert normalized[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert "<tool_call>" not in str(normalized)


def test_consecutive_tool_blocks_wait_for_a_partial_second_block() -> None:
    first = '<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>'
    second = '<tool_call>{"name":"Bash","arguments":{"command":"dir"}}</tool_call>'
    chunks = [
        content_chunk(first + second[:5]),
        content_chunk(second[5:31]),
        content_chunk(second[31:]),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 8}},
    ]

    normalized = normalize_tool_stream(iter(chunks), TOOLS)
    assert next(normalized)["choices"][0]["delta"] == {"role": "assistant"}
    output = list(normalized)
    calls = [
        call
        for chunk in output
        for choice in chunk.get("choices", [])
        for call in choice.get("delta", {}).get("tool_calls", [])
    ]
    assert [call["index"] for call in calls] == [0, 1]
    assert [json.loads(call["function"]["arguments"])["command"] for call in calls] == [
        "pwd", "dir",
    ]
    assert "<tool" not in str(output)
    assert output[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert output[-1] == chunks[-1]


def test_conversion_preserves_delta_and_choice_siblings() -> None:
    chunk = {
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "reasoning_content": "Need the current directory.",
                "content": '<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>',
            },
            "logprobs": {"content": []},
            "finish_reason": None,
        }],
    }

    normalized = list(normalize_tool_stream([chunk], TOOLS))

    preserved = normalized[0]["choices"][0]
    assert normalized[0]["id"] == "chatcmpl-1"
    assert preserved["delta"] == {
        "role": "assistant",
        "reasoning_content": "Need the current directory.",
    }
    assert preserved["logprobs"] == {"content": []}
    assert normalized[-1]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def test_content_bearing_terminal_chunk_gets_tool_finish_reason() -> None:
    chunk = content_chunk(
        '<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>'
    )
    chunk["choices"][0]["finish_reason"] = "stop"
    chunk["usage"] = {"completion_tokens": 4}

    normalized = list(normalize_tool_stream([chunk], TOOLS))

    assert normalized[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert normalized[-1] == {
        "choices": [], "usage": {"completion_tokens": 4},
    }


def test_native_tool_deltas_and_auxiliary_events_are_unchanged() -> None:
    chunks = [
        {"choices": [{
            "index": 0,
            "delta": {"reasoning_content": "need a command"},
            "finish_reason": None,
        }]},
        {"choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "Bash", "arguments": '{"command":'},
            }]},
            "finish_reason": None,
        }]},
        {"choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": '"pwd"}'},
            }]},
            "finish_reason": None,
        }]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"completion_tokens": 3}},
    ]

    assert list(normalize_tool_stream(chunks, TOOLS)) == chunks


def test_malformed_and_unclosed_markup_remains_plain_content() -> None:
    malformed = [
        content_chunk('<tool_call>{"name":"Unknown","arguments":{}}</tool_call>'),
        content_chunk(" then <tool_call>unfinished"),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
    ]

    assert list(normalize_tool_stream(malformed, TOOLS)) == malformed

    usage_after_unclosed = [
        content_chunk("answer <tool_call>unfinished"),
        {"choices": [], "usage": {"completion_tokens": 2}},
    ]
    assert list(normalize_tool_stream(usage_after_unclosed, TOOLS)) == usage_after_unclosed
