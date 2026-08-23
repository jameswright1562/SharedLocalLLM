from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sharedlocalllm_backend.server_engine import (
    ServerEngine,
    build_command,
    server_environment,
    wire_messages,
)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Run a command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    },
}]


class Store:
    def log(self, *_args) -> None:
        return None


class Writer:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_build_command_enables_embedded_mtp_and_rpc() -> None:
    command = build_command(
        Path("llama-server.exe"), model_path="model.gguf", port=8123,
        context=8192, api_key="secret", mtp=True,
        speculation_supported=True, rpc_endpoint="127.0.0.1:5000",
        load_config={
            "flashAttention": True, "useMmap": False, "useMlock": True,
            "cpuThreads": 6, "batchSize": 256,
        },
    )

    assert command[0].endswith("llama-server.exe")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--rpc") + 1] == "127.0.0.1:5000"
    assert command[command.index("--api-key") + 1] == "secret"
    assert command[command.index("-ngl") + 1] == "all"
    assert command[command.index("--split-mode") + 1] == "layer"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert "--no-mmap" in command
    assert "--mlock" in command
    assert command[command.index("--threads") + 1] == "6"
    assert command[command.index("--batch-size") + 1] == "256"
    assert "--no-agent" in command
    assert "--no-ui" in command
    assert "--offline" in command


def test_server_environment_cannot_enable_internal_agent_tools(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_ARG_TOOLS", "all")
    monkeypatch.setenv("LLAMA_ARG_AGENT", "1")
    monkeypatch.setenv("LLAMA_ARG_MCP_SERVERS_JSON", '{"dangerous":true}')
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    environment = server_environment()

    assert "LLAMA_ARG_TOOLS" not in environment
    assert "LLAMA_ARG_AGENT" not in environment
    assert "LLAMA_ARG_MCP_SERVERS_JSON" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_wire_messages_preserves_an_agent_tool_result_round_trip() -> None:
    calls = [{
        "id": "call_1", "type": "function",
        "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
    }]
    messages = [
        {"role": "user", "content": "Where am I?"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "content": "C:\\code", "tool_call_id": "call_1"},
    ]

    assert wire_messages(messages, {"systemPrompt": "Be concise."}) == [
        {"role": "system", "content": "Be concise."},
        messages[0], messages[1], messages[2],
    ]


def test_chat_forwards_tools_and_returns_the_native_tool_message() -> None:
    engine = ServerEngine(Store())
    captured: dict = {}
    call = {
        "id": "call_1", "type": "function",
        "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
    }

    async def read_response(payload, timeout=600):
        captured.update(payload)
        return 200, json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        })

    engine._read_response = read_response  # type: ignore[method-assign]
    result = asyncio.run(engine.chat(
        [{"role": "user", "content": "Run pwd"}], {}, TOOLS, "required"
    ))

    assert captured["tools"] == TOOLS
    assert captured["tool_choice"] == "required"
    assert result["message"]["tool_calls"] == [call]
    assert result["message"]["content"] is None
    assert result["finishReason"] == "tool_calls"
    assert result["usage"]["total_tokens"] == 14


def test_openai_stream_preserves_native_tool_call_argument_fragments() -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
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
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    engine = ServerEngine(Store())
    captured: dict = {}

    async def collect():
        reader = asyncio.StreamReader()
        reader.feed_data(
            f"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{body}".encode()
        )
        reader.feed_eof()
        writer = Writer()

        async def open_request(payload, timeout):
            captured.update(payload)
            return reader, writer

        engine._open_request = open_request  # type: ignore[method-assign]
        streamed = [chunk async for chunk in engine.chat_openai_stream([], {}, TOOLS, "auto")]
        assert writer.closed is True
        return streamed

    streamed = asyncio.run(collect())
    assert streamed == chunks
    assert captured["tools"] == TOOLS
    assert captured["stream_options"] == {"include_usage": True}


def test_openai_stream_converts_text_tool_markup_for_agent_clients() -> None:
    markup = '<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>'
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": markup[:35]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": markup[35:]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    engine = ServerEngine(Store())

    async def collect():
        reader = asyncio.StreamReader()
        reader.feed_data(f"HTTP/1.1 200 OK\r\n\r\n{body}".encode())
        reader.feed_eof()

        async def open_request(_payload, _timeout):
            return reader, Writer()

        engine._open_request = open_request  # type: ignore[method-assign]
        return [chunk async for chunk in engine.chat_openai_stream([], {}, TOOLS, "auto")]

    streamed = asyncio.run(collect())
    assert all("<tool_call>" not in str(chunk) for chunk in streamed)
    assert streamed[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Bash"
    assert streamed[-1]["choices"][0]["finish_reason"] == "tool_calls"
