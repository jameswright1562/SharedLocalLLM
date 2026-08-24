from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from sharedlocalllm_backend.server_engine import (
    ServerEngine,
    build_command,
    relay_server_output,
    server_environment,
    startup_failure_message,
    wire_messages,
    write_model_output,
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
    def __init__(self) -> None:
        self.entries: list[tuple] = []

    def log(self, *_args) -> None:
        self.entries.append(_args)


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
        gpu_layers=48, tensor_split=[20, 28], reasoning_preserve=True,
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
    assert command[command.index("-ngl") + 1] == "48"
    assert command[command.index("--tensor-split") + 1] == "20,28"
    assert command[command.index("--fit") + 1] == "off"
    assert "--reasoning-preserve" in command
    assert command[command.index("--split-mode") + 1] == "layer"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--load-mode") + 1] == "mlock"
    assert "--mmap" not in command
    assert "--no-mmap" not in command
    assert "--mlock" not in command
    assert command[command.index("--threads") + 1] == "6"
    assert command[command.index("--batch-size") + 1] == "256"
    assert "--no-agent" in command
    assert "--no-ui" in command
    assert "--offline" in command


def test_build_command_passes_autotuned_knobs_only_when_set() -> None:
    tuned = build_command(
        Path("llama-server.exe"), model_path="model.gguf", port=8123,
        context=8192, api_key=None, mtp=False, speculation_supported=False,
        rpc_endpoint=None, gpu_layers=None, tensor_split=None,
        reasoning_preserve=False,
        load_config={
            "uBatch": 256, "kvCacheK": "q4_0", "kvCacheV": "q8_0",
            "noOpOffload": 1, "rpcPoll": 50,
        },
    )

    assert tuned[tuned.index("--ubatch-size") + 1] == "256"
    assert tuned[tuned.index("--cache-type-k") + 1] == "q4_0"
    assert tuned[tuned.index("--cache-type-v") + 1] == "q8_0"
    assert "--no-op-offload" in tuned
    assert tuned[tuned.index("--poll") + 1] == "50"


def test_build_command_without_tuned_values_keeps_defaults() -> None:
    plain = build_command(
        Path("llama-server.exe"), model_path="model.gguf", port=8123,
        context=8192, api_key=None, mtp=False, speculation_supported=False,
        rpc_endpoint=None, gpu_layers=None, tensor_split=None,
        reasoning_preserve=False, load_config={},
    )

    for flag in (
        "--ubatch-size", "--cache-type-k", "--cache-type-v",
        "--no-op-offload", "--poll", "-ngl", "--tensor-split",
        "--fit", "--reasoning-preserve",
    ):
        assert flag not in plain


def test_server_environment_ignores_all_external_llama_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_ARG_TOOLS", "all")
    monkeypatch.setenv("LLAMA_ARG_AGENT", "1")
    monkeypatch.setenv("LLAMA_ARG_MCP_SERVERS_JSON", '{"dangerous":true}')
    monkeypatch.setenv("LLAMA_ARG_EMBEDDINGS", "1")
    monkeypatch.setenv("LLAMA_ARG_PORT", "11434")
    monkeypatch.setenv("LLAMA_ARG_RERANKING", "1")
    monkeypatch.setenv("LLAMA_API_KEY", "wrong-secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    environment = server_environment()

    assert not any(name.startswith("LLAMA_ARG_") for name in environment)
    assert "LLAMA_API_KEY" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_server_output_is_redacted_and_tee_d_to_the_console(
    tmp_path: Path, capsys,
) -> None:
    model_path = r"C:\Users\James\.cache\models\qwen.gguf"
    source = io.BytesIO(
        (
            f"loading model '{model_path}' with key top-secret\n"
            "CUDA Graph id 1378 reused\n"
            "create_tensor: loading tensor blk.15.attn_v.input_scale\n"
            "missing result_norm/result_embd tensor\n"
        ).encode()
    )
    log_path = tmp_path / "llama-server.log"

    relay_server_output(source, log_path, model_path, "top-secret")

    console = capsys.readouterr().err
    saved = log_path.read_text(encoding="utf-8")
    for output in (console, saved):
        assert "qwen.gguf" not in output
        assert "top-secret" not in output
        assert "<model-path>" in output
        assert "<redacted>" in output
        assert "missing result_norm/result_embd tensor" in output
    assert "CUDA Graph id" not in console
    assert "create_tensor: loading tensor" not in console
    assert "CUDA Graph id 1378 reused" in saved
    assert "create_tensor: loading tensor blk.15.attn_v.input_scale" in saved


def test_known_mtp_embedding_assertion_has_an_actionable_summary() -> None:
    message = startup_failure_message(
        "embeddings enabled\nGGML_ASSERT missing result_norm/result_embd tensor failed"
    )

    assert "embedding or reranking mode" in message
    assert "MTP" in message
    assert "Restart SharedLocalLLM" in message


def test_model_output_is_printed_in_full_with_sensitive_values_redacted(capsys) -> None:
    model_path = r"D:\private-models\qwen.gguf"

    write_model_output(
        f"First line\nModel is at {model_path}; key is top-secret.\nLast line",
        model_path=model_path,
        api_key="top-secret",
    )

    console = capsys.readouterr().err
    assert "[model response #" in console
    assert "First line\nModel is at <model-path>; key is <redacted>.\nLast line" in console
    assert model_path not in console
    assert "top-secret" not in console
    assert "[/model response #" in console


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


def test_chat_forwards_tools_and_returns_the_native_tool_message(capsys) -> None:
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
    console = capsys.readouterr().err
    assert "<no text output>" in console
    assert '"command":"pwd"' not in console


def test_buffered_generation_read_has_no_response_deadline() -> None:
    engine = ServerEngine(Store())

    async def request():
        reader = asyncio.StreamReader()
        reader.feed_data(b"HTTP/1.1 200 OK\r\n\r\n{\"choices\": []}")
        reader.feed_eof()
        writer = Writer()

        async def open_request(_payload, timeout):
            assert timeout is None
            return reader, writer

        engine._open_request = open_request  # type: ignore[method-assign]
        response = await engine._read_response({})
        assert writer.closed is True
        return response

    assert asyncio.run(request()) == (200, '{"choices": []}')


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


def test_openai_stream_prints_the_complete_text_response(capsys) -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "world."}, "finish_reason": None}]},
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
        return [chunk async for chunk in engine.chat_openai_stream([], {})]

    assert asyncio.run(collect()) == chunks
    console = capsys.readouterr().err
    assert "Hello world." in console
    assert console.count("[model response #") == 1


def test_ui_stream_prints_the_complete_text_response(capsys) -> None:
    engine = ServerEngine(Store())

    async def sse_events(_payload):
        yield {"choices": [{"delta": {"content": "One "}}]}
        yield {"choices": [{"delta": {"content": "answer"}}]}
        yield {"choices": [], "usage": {"completion_tokens": 2}}

    engine._sse_events = sse_events  # type: ignore[method-assign]

    async def collect():
        return [event async for event in engine.chat_stream([], {})]

    events = asyncio.run(collect())
    assert [event.get("content") for event in events if event["type"] == "token"] == [
        "One ", "answer",
    ]
    console = capsys.readouterr().err
    assert "One answer" in console
    assert console.count("[model response #") == 1


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


def test_openai_tool_stream_yields_answer_text_before_sse_completion() -> None:
    engine = ServerEngine(Store())
    release = asyncio.Event()

    async def sse_events(_payload):
        yield {
            "choices": [{
                "index": 0, "delta": {"content": "early"}, "finish_reason": None,
            }],
        }
        await release.wait()
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    engine._sse_events = sse_events  # type: ignore[method-assign]

    async def consume() -> dict:
        stream = engine.chat_openai_stream([], {}, TOOLS, "required")
        try:
            first = await asyncio.wait_for(anext(stream), 1)
            release.set()
            return first
        finally:
            release.set()
            await stream.aclose()  # type: ignore[attr-defined]

    first = asyncio.run(consume())
    assert first["choices"][0]["delta"]["content"] == "early"
