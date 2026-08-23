from __future__ import annotations

import asyncio
import os

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.inference import InferenceEngine, build_llama_kwargs, layer_totals


def test_defaults_match_previous_behaviour() -> None:
    kwargs = build_llama_kwargs({}, "model.gguf", 8192, 16, None)
    cores = os.cpu_count() or 8
    assert kwargs["model_path"] == "model.gguf"
    assert kwargs["n_ctx"] == 8192
    assert kwargs["n_gpu_layers"] == 16
    assert kwargs["n_threads"] == max(1, cores // 2)
    assert kwargs["n_threads_batch"] == max(1, cores)
    assert kwargs["n_batch"] == 512
    assert kwargs["flash_attn"] is False
    assert kwargs["use_mmap"] is True
    assert kwargs["use_mlock"] is False
    assert kwargs["verbose"] is True


def test_zero_gpu_layers_offloads_automatically() -> None:
    kwargs = build_llama_kwargs({}, "model.gguf", 4096, 0, None)
    assert kwargs["n_gpu_layers"] == -1


def test_validated_cpu_only_load_does_not_auto_offload() -> None:
    kwargs = build_llama_kwargs(
        {"automaticGpuOffload": False}, "model.gguf", 4096, 0, None
    )
    assert kwargs["n_gpu_layers"] == 0


def test_options_flow_through_to_llama() -> None:
    kwargs = build_llama_kwargs(
        {
            "flashAttention": True,
            "useMmap": False,
            "useMlock": True,
            "cpuThreads": 4,
            "batchSize": 2048,
        },
        "model.gguf", 4096, 8, [16.0, 16.0],
    )
    assert kwargs["flash_attn"] is True
    assert kwargs["use_mmap"] is False
    assert kwargs["use_mlock"] is True
    assert kwargs["n_threads"] == 4
    assert kwargs["n_batch"] == 2048
    assert kwargs["tensor_split"] == [16.0, 16.0]


def test_zero_threads_keeps_automatic_default() -> None:
    kwargs = build_llama_kwargs({"cpuThreads": 0}, "model.gguf", 4096, 0, None)
    cores = os.cpu_count() or 8
    assert kwargs["n_threads"] == max(1, cores // 2)


def test_batch_size_is_clamped_to_at_least_one() -> None:
    kwargs = build_llama_kwargs({"batchSize": 0}, "model.gguf", 4096, 0, None)
    assert kwargs["n_batch"] == 1


def test_layer_totals_split_remote_gpu_cpu_and_local() -> None:
    allocations = [
        {"nodeId": "peer", "layers": 12},
        {"nodeId": "peer", "layers": 4, "kind": "cpu"},
        {"nodeId": "local", "layers": 16},
    ]
    assert layer_totals(allocations, "peer", "local", True) == (12, 4, 16)


def test_layer_totals_default_all_allocations_to_gpu() -> None:
    allocations = [
        {"nodeId": "peer", "layers": 8},
        {"nodeId": "local", "layers": 8},
    ]
    assert layer_totals(allocations, "peer", "local", True) == (8, 0, 8)


def test_layer_totals_ignore_cpu_without_flag() -> None:
    allocations = [
        {"nodeId": "peer", "layers": 8, "kind": "cpu"},
    ]
    assert layer_totals(allocations, "peer", "local") == (8, 0, 0)


def test_layer_totals_ignore_unknown_nodes_and_missing_peer() -> None:
    allocations = [
        {"nodeId": "peer", "layers": 8},
        {"nodeId": "stranger", "layers": 99},
    ]
    assert layer_totals(allocations, None, "local", True) == (0, 0, 0)


def test_wire_messages_preserves_tool_calls_and_tool_results() -> None:
    engine = InferenceEngine(None)
    tool_calls = [{
        "id": "call_1", "type": "function",
        "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
    }]
    messages = [
        {
            "role": "assistant", "content": None,
            "reasoning_content": "Need the current directory.", "tool_calls": tool_calls,
        },
        {"role": "tool", "content": "C:\\code", "tool_call_id": "call_1"},
    ]

    assert engine._wire_messages(messages, {}) == messages


def test_chat_sync_passes_tools_and_preserves_native_tool_response() -> None:
    tool_calls = [{
        "id": "call_1", "type": "function",
        "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
    }]
    tools = [{
        "type": "function",
        "function": {"name": "Bash", "parameters": {"type": "object"}},
    }]

    class FakeLlama:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        def create_chat_completion(self, **kwargs):
            self.kwargs = kwargs
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            }

        def tokenize(self, text: bytes, add_bos: bool = False):
            return [1]

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()
    result = engine._chat_sync(
        [{"role": "user", "content": "Run pwd"}],
        {"temperature": 0.1, "maxTokens": 64},
        tools,
        "auto",
    )

    assert engine.llm.kwargs["tools"] == tools
    assert engine.llm.kwargs["tool_choice"] == "auto"
    assert result["message"]["tool_calls"] == tool_calls
    assert result["message"]["content"] is None
    assert result["finishReason"] == "tool_calls"
    assert result["usage"]["total_tokens"] == 20


def test_openai_stream_preserves_native_tool_call_chunks() -> None:
    tools = [{
        "type": "function",
        "function": {"name": "Bash", "parameters": {"type": "object"}},
    }]
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
            }]},
            "finish_reason": None,
        }]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]

    class FakeLlama:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        def create_chat_completion(self, **kwargs):
            self.kwargs = kwargs
            return iter(chunks)

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()

    async def collect() -> list[dict]:
        return [
            chunk async for chunk in engine.chat_openai_stream(
                [{"role": "user", "content": "Run pwd"}], {}, tools, "auto"
            )
        ]

    assert asyncio.run(collect()) == chunks
    assert engine.llm.kwargs["tools"] == tools
    assert engine.llm.kwargs["tool_choice"] == "auto"


def test_chat_sync_converts_qwen_text_tool_markup() -> None:
    tools = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]

    class FakeLlama:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<tool_call>\n<function=Bash>\n<parameter=command>\n"
                            "Get-Location\n</parameter>\n</function>\n</tool_call>"
                        ),
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 10},
            }

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()
    result = engine._chat_sync([], {}, tools, "auto")
    assert result["message"]["content"] is None
    assert result["message"]["tool_calls"][0]["function"]["name"] == "Bash"
    assert result["finishReason"] == "tool_calls"


def test_openai_stream_converts_qwen_text_tool_markup() -> None:
    tools = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]

    class FakeLlama:
        def create_chat_completion(self, **_kwargs):
            return iter([
                {"choices": [{
                    "index": 0,
                    "delta": {"content": "<tool_call>{\"name\":\"Bash\","},
                    "finish_reason": None,
                }]},
                {"choices": [{
                    "index": 0,
                    "delta": {"content": "\"arguments\":{\"command\":\"pwd\"}}</tool_call>"},
                    "finish_reason": None,
                }]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ])

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()

    async def collect() -> list[dict]:
        return [chunk async for chunk in engine.chat_openai_stream([], {}, tools, "auto")]

    chunks = asyncio.run(collect())
    assert all("<tool_call>" not in str(chunk) for chunk in chunks)
    assert chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Bash"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_keeps_tool_markup_as_plain_content() -> None:
    tools = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]

    class FakeLlama:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"Bash","arguments":{}}</tool_call>',
                    },
                    "finish_reason": "stop",
                }],
                "usage": {},
            }

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()
    result = engine._chat_sync([], {}, tools, "none")
    assert "tool_calls" not in result["message"]
    assert "<tool_call>" in result["message"]["content"]
    assert result["finishReason"] == "stop"


def test_openai_stream_separates_reasoning_from_answer_content() -> None:
    class FakeLlama:
        def create_chat_completion(self, **_kwargs):
            return iter([
                {"choices": [{
                    "index": 0, "delta": {"content": "<think>secret"},
                    "finish_reason": None,
                }]},
                {"choices": [{
                    "index": 0, "delta": {"content": " plan</think>answer"},
                    "finish_reason": None,
                }]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ])

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()
    engine.reasoning = True

    async def collect() -> list[dict]:
        return [chunk async for chunk in engine.chat_openai_stream([], {})]

    chunks = asyncio.run(collect())
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
    content = "".join(delta.get("content", "") for delta in deltas)
    assert reasoning == "secret plan"
    assert content == "answer"
    assert "<think>" not in str(chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_retries_qwen_template_with_mapping_arguments() -> None:
    class FakeLlama:
        def __init__(self) -> None:
            self.arguments: list[object] = []

        def create_chat_completion(self, **kwargs):
            arguments = kwargs["messages"][0]["tool_calls"][0]["function"]["arguments"]
            self.arguments.append(arguments)
            if isinstance(arguments, str):
                raise TypeError("Can only get item pairs from a mapping.")
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": "continued"},
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 1},
            }

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()
    messages = [{
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
        }],
    }]
    result = engine._chat_sync(messages, {}, [], "auto")
    assert engine.llm.arguments == ['{"command":"pwd"}', {"command": "pwd"}]
    assert result["content"] == "continued"


def test_chat_reports_an_actionable_incompatible_template_error() -> None:
    class FakeLlama:
        def create_chat_completion(self, **_kwargs):
            raise TypeError("Can only get item pairs from a mapping.")

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()
    with pytest.raises(BackendError, match="chat template") as excinfo:
        engine._chat_sync([{"role": "user", "content": "hi"}], {}, [], "auto")
    assert excinfo.value.code == "chat_template_tools_invalid"


def test_chat_stream_sets_cancel_when_the_consumer_disconnects() -> None:
    class FakeLlama:
        def create_chat_completion(self, **_kwargs):
            for index in range(500):
                yield {
                    "choices": [{
                        "index": 0, "delta": {"content": f"t{index}"},
                        "finish_reason": None,
                    }]
                }

        def tokenize(self, text: bytes, add_bos: bool = False):
            return [1]

    engine = InferenceEngine(None)
    engine.llm = FakeLlama()

    async def consume() -> None:
        stream = engine.chat_stream([{"role": "user", "content": "hi"}], {})
        async for _chunk in stream:
            break
        await stream.aclose()

    asyncio.run(consume())
    assert engine._cancel.is_set()


def test_forced_combined_gpu_launch_without_split_offloads_automatically(monkeypatch) -> None:
    class FakeForwarder:
        def __init__(self, peer, model_id=None, include_cpu=False) -> None:
            return None

        async def start(self) -> str:
            return "127.0.0.1:5000"

        async def stop(self) -> None:
            return None

    class FakeStore:
        def log(self, *_args) -> None:
            return None

    captured: dict = {}

    def fake_load_sync(self, path, context, gpu_layers, tensor_split, load_config):
        captured["gpu_layers"] = gpu_layers
        captured["load_config"] = load_config

    monkeypatch.setattr("sharedlocalllm_backend.inference.RpcForwarder", FakeForwarder)
    monkeypatch.setattr(
        "sharedlocalllm_backend.inference.prepare_rpc_load", lambda *_args: [1.0]
    )
    monkeypatch.setattr(InferenceEngine, "_load_sync", fake_load_sync)

    engine = InferenceEngine(FakeStore())
    model = {"id": "model", "name": "Model", "fit": "combined-gpu"}
    asyncio.run(engine.load(
        model,
        "model.gguf",
        {"contextSize": 4096, "gpuLayers": [], "automaticGpuOffload": False},
        peer=object(),
        local_id="local",
        peer_id="remote",
    ))

    kwargs = build_llama_kwargs(
        captured["load_config"], "model.gguf", 4096, captured["gpu_layers"], None
    )
    assert kwargs["n_gpu_layers"] == -1
