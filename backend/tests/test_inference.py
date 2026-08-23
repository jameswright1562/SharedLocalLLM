from __future__ import annotations

import asyncio
import os

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
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
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
