from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import threading
import time
from typing import Any, AsyncIterator

import numpy as np
import numpy.typing as npt

from .errors import BackendError
from .openai_compat import reasoning_stream_chunks, sampling_kwargs
from .peer import RpcForwarder
from .reasoning import ReasoningStreamSplitter, is_reasoning_model, split_reasoning
from .rpc_native import NativeRpcServer, prepare_rpc_load
from .tool_calls import normalize_tool_message, normalize_tool_stream, template_tool_inputs

LLAMA_SPLIT_MODE_LAYER = 1

# Stable ggml type enum ids (llama-cpp-python's type_k/type_v take ints).
KV_CACHE_TYPE_IDS = {
    "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8,
}


def _publish_thread_event(
    loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[Any], value: Any,
) -> None:
    """Publish from llama's daemon thread unless its consumer loop has closed."""
    if loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(queue.put_nowait, value)
    except RuntimeError:
        pass


def _kv_cache_type_id(value: Any) -> int | None:
    if not value:
        return None
    return KV_CACHE_TYPE_IDS.get(str(value).strip().lower())


@contextlib.contextmanager
def unified_kv_defaults() -> Any:
    """Force ``kv_unified=True`` while llama-cpp-python builds its context.

    llama-cpp-python 0.3.x constructs ``llama_context_params`` from the native
    default, which ships with the unified KV buffer disabled, and exposes no
    constructor override. The low-level module attribute it reads at call time
    is therefore wrapped for the duration of one ``Llama(...)`` construction.
    Only relevant when serving several parallel sequences; a single-sequence
    context behaves identically either way.
    """
    import llama_cpp.llama_cpp as lowlevel

    original = lowlevel.llama_context_default_params

    def unified() -> Any:
        params = original()
        params.kv_unified = True
        return params

    lowlevel.llama_context_default_params = unified  # type: ignore[assignment]
    try:
        yield
    finally:
        lowlevel.llama_context_default_params = original  # type: ignore[assignment]


def build_llama_kwargs(
    load_config: dict[str, Any], path: str, context: int, gpu_layers: int,
    tensor_split: list[float] | None,
) -> dict[str, Any]:
    """Map a model load config into llama-cpp-python constructor arguments.

    Unknown or unset options fall back to the previous hardcoded behaviour:
    automatic CPU threads, a 512-token batch, mmap enabled, and standard (non
    flash) attention. ``cpuThreads=0`` keeps the automatic half-core default.
    Autotuned extras (``uBatch``, ``kvCacheK``, ``kvCacheV``) are applied only
    when present; RPC-only knobs like ``--poll`` have no equivalent here.
    """
    cores = os.cpu_count() or 8
    requested_threads = int(load_config.get("cpuThreads", 0))
    batch = max(1, int(load_config.get("batchSize", 512)))
    kwargs: dict[str, Any] = {
        "model_path": path,
        "n_ctx": context,
        "n_gpu_layers": (
            gpu_layers
            if gpu_layers > 0
            else (-1 if load_config.get("automaticGpuOffload", True) else 0)
        ),
        "split_mode": LLAMA_SPLIT_MODE_LAYER,
        "tensor_split": tensor_split,
        "n_threads": max(1, requested_threads) if requested_threads > 0 else max(1, cores // 2),
        "n_threads_batch": max(1, cores),
        "n_batch": batch,
        "flash_attn": bool(load_config.get("flashAttention", False)),
        "use_mmap": bool(load_config.get("useMmap", True)),
        "use_mlock": bool(load_config.get("useMlock", False)),
        "verbose": True,
    }
    raw_ubatch = load_config.get("uBatch")
    if raw_ubatch:
        kwargs["n_ubatch"] = max(1, min(int(raw_ubatch), batch))
    k_type = _kv_cache_type_id(load_config.get("kvCacheK"))
    v_type = _kv_cache_type_id(load_config.get("kvCacheV"))
    if k_type is not None:
        kwargs["type_k"] = k_type
    if v_type is not None:
        kwargs["type_v"] = v_type
    return kwargs


def layer_totals(
    allocations: list[dict[str, Any]], peer_id: str | None, local_id: str,
    include_remote_cpu: bool = False,
) -> tuple[int, int, int]:
    """Sum layer allocations into (remote GPU, remote CPU, local) totals.

    A remote allocation with ``kind == "cpu"`` targets the worker's CPU over
    RPC, but only when ``include_remote_cpu`` is set; otherwise every allocation
    counts as GPU layers. Local CPU offload stays implicit in llama.cpp (layers
    not assigned to any device).
    """
    remote_gpu = remote_cpu = local = 0
    for allocation in allocations:
        node_id = allocation.get("nodeId")
        layers = int(allocation.get("layers", 0))
        if peer_id and node_id == peer_id:
            if include_remote_cpu and allocation.get("kind") == "cpu":
                remote_cpu += layers
            else:
                remote_gpu += layers
        elif node_id == local_id:
            local += layers
    return remote_gpu, remote_cpu, local


class InferenceEngine:
    def __init__(self, store: Any) -> None:
        self.store = store
        self.llm: Any = None
        self.model_id: str | None = None
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()
        self._cancel = threading.Event()
        self._forwarder: RpcForwarder | None = None
        self._rpc_worker = NativeRpcServer()
        self.reasoning = False

    async def load(
        self, model: dict[str, Any], path: str, load_config: dict[str, Any],
        peer: Any, local_id: str, peer_id: str | None,
    ) -> None:
        async with self._async_lock:
            await self._unload_locked()
            allocations = load_config.get("gpuLayers") or []
            rpc_endpoint = None
            include_remote_cpu = bool(load_config.get("includeRemoteCpu"))
            remote_gpu_layers, remote_cpu_layers, local_layers = layer_totals(
                allocations, peer_id, local_id, include_remote_cpu
            )
            remote_total = remote_gpu_layers + remote_cpu_layers
            if peer_id and (remote_total > 0 or model.get("fit") == "combined-gpu"):
                self._forwarder = RpcForwarder(
                    peer, model_id=model["id"], include_cpu=remote_cpu_layers > 0
                )
                rpc_endpoint = await self._forwarder.start()
            total_layers = remote_gpu_layers + remote_cpu_layers + local_layers
            if rpc_endpoint and total_layers <= 0:
                # Forced launch of a combined-GPU model without any usable
                # split: stage the worker devices, then let llama.cpp spread
                # layers evenly across them instead of pretending two layers
                # exist (which silently degraded the model to near-pure CPU).
                load_config = {**load_config, "automaticGpuOffload": True}
                await asyncio.to_thread(prepare_rpc_load, rpc_endpoint, 0, 0, 0)
                tensor_split = None
            else:
                tensor_split = await asyncio.to_thread(
                    prepare_rpc_load, rpc_endpoint, remote_gpu_layers, remote_cpu_layers,
                    local_layers,
                )
            context = max(512, int(load_config.get("contextSize", 4096)))
            self.store.log(
                "INFO", "model_load",
                f"model={model['name']} ctx={context} rpc={rpc_endpoint} split={tensor_split} "
                f"remote_gpu={remote_gpu_layers} remote_cpu={remote_cpu_layers} local={local_layers} "
                f"flash_attn={bool(load_config.get('flashAttention', False))} "
                f"mmap={bool(load_config.get('useMmap', True))} "
                f"mlock={bool(load_config.get('useMlock', False))} "
                f"threads={int(load_config.get('cpuThreads', 0))} "
                f"batch={int(load_config.get('batchSize', 512))}",
            )
            try:
                await asyncio.to_thread(
                    self._load_sync, path, context, total_layers, tensor_split, load_config
                )
            except Exception as error:
                await self._stop_forwarder()
                raise BackendError(
                    "model_load_failed", str(error),
                    "Reduce context/model size or review the CUDA/RPC logs on both computers.",
                ) from error
            self.model_id = model["id"]
            self.reasoning = is_reasoning_model(model["name"])
            self.store.log("INFO", "model_loaded", model["name"])

    def _load_sync(
        self, path: str, context: int, gpu_layers: int, tensor_split: list[float] | None,
        load_config: dict[str, Any],
    ) -> None:
        from llama_cpp import Llama

        kwargs = build_llama_kwargs(load_config, path, context, gpu_layers, tensor_split)
        with self._sync_lock:
            if load_config.get("kvUnified"):
                with unified_kv_defaults():
                    self.llm = Llama(**kwargs)
            else:
                self.llm = Llama(**kwargs)

    async def unload(self) -> None:
        async with self._async_lock:
            await self._unload_locked()

    async def _unload_locked(self) -> None:
        await asyncio.to_thread(self._drop_sync)
        if self.model_id:
            self.store.log("INFO", "model_unloaded", self.model_id)
        self.model_id = None
        await self._stop_forwarder()

    def _drop_sync(self) -> None:
        with self._sync_lock:
            self.llm = None
            gc.collect()

    async def _stop_forwarder(self) -> None:
        if self._forwarder:
            await self._forwarder.stop()
            self._forwarder = None

    async def open_rpc_worker_connection(
        self, include_cpu: bool = False
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        endpoint = await asyncio.to_thread(self._rpc_worker.start, include_cpu)
        cache_state = self._rpc_worker.cache_dir or "off"
        self.store.log("INFO", "rpc_worker_ready", f"{endpoint} include_cpu={include_cpu} cache={cache_state}")
        host, port = endpoint.split(":", 1)
        return await asyncio.open_connection(host, int(port))

    async def chat(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str],
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
    ) -> dict[str, Any]:
        if images:
            raise BackendError(
                "vision_not_migrated", "Vision attachments are not enabled in the Python migration yet.",
                "Use text-only chat on this branch for now."
            )
        async with self._async_lock:
            if not self.llm:
                raise BackendError("model_not_loaded", "Start a model before chatting.")
            self._cancel.clear()
            return await asyncio.to_thread(
                self._chat_sync, messages, settings, tools, tool_choice
            )

    async def chat_openai_stream(
        self, messages: list[dict[str, Any]], settings: dict[str, Any],
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Relay llama-cpp-python's OpenAI chunks without losing tool calls."""
        from llama_cpp import LogitsProcessorList

        async with self._async_lock:
            if not self.llm:
                raise BackendError("model_not_loaded", "Start a model before chatting.")
            self._cancel.clear()
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()

            def run() -> None:
                prompt_tokens = 0
                completion_tokens = 0
                pass_prompt_tokens: int | None = None
                evaluated_tokens: int | None = None
                finish_reason: str | None = None
                usage_seen = False
                completed = False

                def count_tokens(
                    tokens: npt.NDArray[np.intc], logits: npt.NDArray[np.single],
                ) -> npt.NDArray[np.single]:
                    nonlocal prompt_tokens, completion_tokens
                    nonlocal pass_prompt_tokens, evaluated_tokens
                    if self._cancel.is_set():
                        raise InterruptedError
                    count = len(tokens)
                    # A discontinuity starts another internal completion pass,
                    # as used by some tool-oriented llama-cpp chat handlers.
                    if evaluated_tokens is None or count != evaluated_tokens + 1:
                        if pass_prompt_tokens is not None and evaluated_tokens is not None:
                            completion_tokens += max(
                                0, evaluated_tokens - pass_prompt_tokens
                            )
                        prompt_tokens += count
                        pass_prompt_tokens = count
                    evaluated_tokens = count
                    return logits

                try:
                    with self._sync_lock:
                        chunks = self._create_chat_completion(
                            messages=self._wire_messages(messages, settings),
                            temperature=float(settings.get("temperature", 0.7)),
                            max_tokens=int(settings.get("maxTokens", 512)),
                            **sampling_kwargs(settings),
                            tools=tools,
                            tool_choice=tool_choice,
                            logits_processor=LogitsProcessorList([count_tokens]),
                            stream=True,
                        )
                        needs_tool_fallback = bool(
                            tools and tool_choice != "none" and not isinstance(tool_choice, dict)
                        )

                        def native_values():
                            nonlocal finish_reason, usage_seen, completed
                            for chunk in chunks:
                                if self._cancel.is_set():
                                    break
                                value = dict(chunk)
                                usage_seen = usage_seen or isinstance(value.get("usage"), dict)
                                for choice in value.get("choices", []):
                                    if isinstance(choice, dict) and choice.get("finish_reason"):
                                        finish_reason = str(choice["finish_reason"])
                                yield value
                            else:
                                completed = True

                        values = native_values()
                        if needs_tool_fallback and tools:
                            values = normalize_tool_stream(values, tools)
                        for value in reasoning_stream_chunks(values, self.reasoning):
                            _publish_thread_event(loop, queue, value)
                        if (
                            completed and not usage_seen and pass_prompt_tokens is not None
                            and evaluated_tokens is not None
                        ):
                            completion_tokens += max(
                                0, evaluated_tokens - pass_prompt_tokens
                            )
                            if finish_reason == "length":
                                completion_tokens += 1
                            usage = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            }
                            _publish_thread_event(
                                loop, queue, {"choices": [], "usage": usage}
                            )
                except BaseException as error:
                    _publish_thread_event(loop, queue, error)
                finally:
                    _publish_thread_event(loop, queue, None)

            threading.Thread(target=run, name="llama-openai-stream", daemon=True).start()
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise BackendError("generation_failed", str(item)) from item
                    yield item
            finally:
                self._cancel.set()

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        """Retry mapping-oriented GGUF templates with decoded tool arguments."""
        mapping_error = "Can only get item pairs from a mapping."
        try:
            return self.llm.create_chat_completion(**kwargs)
        except TypeError as error:
            if mapping_error not in str(error):
                raise
        messages, tools = template_tool_inputs(
            kwargs.get("messages", []), kwargs.get("tools")
        )
        retry = {**kwargs, "messages": messages, "tools": tools}
        if messages != kwargs.get("messages", []) or tools != kwargs.get("tools"):
            try:
                return self.llm.create_chat_completion(**retry)
            except TypeError as error:
                if mapping_error not in str(error):
                    raise
        raise BackendError(
            "chat_template_tools_invalid",
            "The model's embedded chat template rejected the OpenAI tool history or schema.",
            "Use a GGUF with a tool-compatible chat template.",
        )

    async def chat_stream(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        if images:
            raise BackendError(
                "vision_not_migrated", "Vision attachments are not enabled in the Python migration yet.",
                "Use text-only chat on this branch for now."
            )
        async with self._async_lock:
            if not self.llm:
                raise BackendError("model_not_loaded", "Start a model before chatting.")
            self._cancel.clear()
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict[str, Any] | BaseException | None] = asyncio.Queue()

            def run() -> None:
                try:
                    wire = self._wire_messages(messages, settings)
                    splitter = ReasoningStreamSplitter(self.reasoning)
                    reasoning_parts: list[str] = []
                    answer_parts: list[str] = []
                    started: float | None = None
                    with self._sync_lock:
                        chunks = self.llm.create_chat_completion(
                            messages=wire,
                            temperature=float(settings.get("temperature", 0.7)),
                            max_tokens=int(settings.get("maxTokens", 512)),
                            **sampling_kwargs(settings),
                            stream=True,
                        )
                        for chunk in chunks:
                            if self._cancel.is_set():
                                break
                            content = chunk["choices"][0].get("delta", {}).get("content")
                            if content:
                                if started is None:
                                    started = time.monotonic()
                                for kind, piece in splitter.push(str(content)):
                                    (reasoning_parts if kind == "reasoning" else answer_parts).append(piece)
                                    _publish_thread_event(
                                        loop, queue, {"type": kind, "content": piece}
                                    )
                    for kind, piece in splitter.finish():
                        (reasoning_parts if kind == "reasoning" else answer_parts).append(piece)
                        _publish_thread_event(loop, queue, {"type": kind, "content": piece})
                    full = "".join(reasoning_parts) + "".join(answer_parts)
                    completion_tokens = 1
                    with self._sync_lock:
                        completion_tokens = max(
                            1, len(self.llm.tokenize(full.encode("utf-8"), add_bos=False))
                        )
                    elapsed = max((time.monotonic() - started) if started else 0.001, 0.001)
                    _publish_thread_event(
                        loop,
                        queue,
                        {"type": "stats", "tokensPerSecond": round(completion_tokens / elapsed, 2)},
                    )
                    _publish_thread_event(loop, queue, {"type": "done"})
                except BaseException as error:
                    _publish_thread_event(loop, queue, error)
                finally:
                    _publish_thread_event(loop, queue, None)

            thread = threading.Thread(target=run, name="llama-chat-stream", daemon=True)
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise BackendError("generation_failed", str(item)) from item
                    yield item
            finally:
                # A consumer that disconnects (GeneratorExit) must stop the
                # producer; otherwise generation runs to max_tokens while
                # holding _sync_lock and blocks chats, benchmarks, and unload.
                self._cancel.set()

    def _wire_messages(
        self, messages: list[dict[str, Any]], settings: dict[str, Any]
    ) -> list[dict[str, Any]]:
        allowed = (
            "role", "content", "reasoning_content", "tool_calls",
            "tool_call_id", "function_call", "name",
        )
        wire = [
            {key: item[key] for key in allowed if key in item}
            for item in messages
        ]
        system = settings.get("systemPrompt", "").strip()
        if system and not any(item["role"] == "system" for item in wire):
            wire.insert(0, {"role": "system", "content": system})
        return wire

    def _chat_sync(
        self, messages: list[dict[str, Any]], settings: dict[str, Any],
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
    ) -> dict[str, Any]:
        from llama_cpp import LogitsProcessorList

        wire = self._wire_messages(messages, settings)

        def abort_on_cancel(
            _tokens: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
        ) -> npt.NDArray[np.single]:
            if self._cancel.is_set():
                raise InterruptedError
            return logits

        started = time.monotonic()
        with self._sync_lock:
            try:
                result = self._create_chat_completion(
                    messages=wire,
                    temperature=float(settings.get("temperature", 0.7)),
                    max_tokens=int(settings.get("maxTokens", 512)),
                    **sampling_kwargs(settings),
                    logits_processor=LogitsProcessorList([abort_on_cancel]),
                    tools=tools,
                    tool_choice=tool_choice,
                )
            except InterruptedError:
                self.llm.reset()
                return {"content": "", "reasoning": "", "tokensPerSecond": 0.0}
        elapsed = max(time.monotonic() - started, 0.001)
        usage = result.get("usage") or {}
        completion = max(1, int(usage.get("completion_tokens") or 1))
        choice = result["choices"][0]
        message = dict(choice["message"])
        message, finish_reason = normalize_tool_message(
            message, choice.get("finish_reason"), tools if tool_choice != "none" else None
        )
        native_content = message.get("content")
        content = str(native_content or "")
        reasoning, answer = split_reasoning(content, self.reasoning)
        message["content"] = None if native_content is None else answer
        if reasoning:
            message["reasoning_content"] = reasoning
        return {
            "content": answer,
            "reasoning": reasoning,
            "tokensPerSecond": round(completion / elapsed, 2),
            "message": message,
            "finishReason": finish_reason,
            "usage": usage,
        }

    def cancel(self) -> None:
        self._cancel.set()

    async def benchmark(self) -> tuple[float, float]:
        async with self._async_lock:
            if not self.llm:
                raise BackendError("model_not_loaded", "Start the model before benchmarking it.")
            return await asyncio.to_thread(self._benchmark_sync)

    def _benchmark_sync(self) -> tuple[float, float]:
        from llama_cpp import LogitsProcessorList

        prompt = "SharedLocalLLM benchmark: explain why local inference is useful in one paragraph."
        with self._sync_lock:
            if self._cancel.is_set():
                raise BackendError("generation_cancelled", "The benchmark was cancelled.")
            tokens = self.llm.tokenize(prompt.encode(), add_bos=True)
            self.llm.reset()
            started = time.perf_counter()
            self.llm.eval(tokens)
            prompt_rate = len(tokens) / max(time.perf_counter() - started, 0.001)

            def abort_on_cancel(
                _tokens: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
            ) -> npt.NDArray[np.single]:
                if self._cancel.is_set():
                    raise InterruptedError
                return logits

            try:
                started = time.perf_counter()
                result = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=64,
                    temperature=0.0,
                    logits_processor=LogitsProcessorList([abort_on_cancel]),
                )
            except InterruptedError:
                self.llm.reset()
                raise BackendError("generation_cancelled", "The benchmark was cancelled.") from None
            elapsed = max(time.perf_counter() - started, 0.001)
            text = str(result["choices"][0].get("text") or "")
            generated = max(1, len(self.llm.tokenize(text.encode(), add_bos=False)))
            return prompt_rate, generated / elapsed
