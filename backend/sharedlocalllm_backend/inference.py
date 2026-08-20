from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
from typing import Any, AsyncIterator

from .errors import BackendError
from .peer import RpcForwarder
from .reasoning import ReasoningStreamSplitter, is_reasoning_model, split_reasoning
from .rpc_native import NativeRpcServer, prepare_rpc_load


def build_llama_kwargs(
    load_config: dict[str, Any], path: str, context: int, gpu_layers: int,
    tensor_split: list[float] | None,
) -> dict[str, Any]:
    """Map a model load config into llama-cpp-python constructor arguments.

    Unknown or unset options fall back to the previous hardcoded behaviour:
    automatic CPU threads, a 512-token batch, mmap enabled, and standard (non
    flash) attention. ``cpuThreads=0`` keeps the automatic half-core default.
    """
    from llama_cpp import LLAMA_SPLIT_MODE_LAYER

    cores = os.cpu_count() or 8
    requested_threads = int(load_config.get("cpuThreads", 0))
    return {
        "model_path": path,
        "n_ctx": context,
        "n_gpu_layers": gpu_layers if gpu_layers > 0 else -1,
        "split_mode": LLAMA_SPLIT_MODE_LAYER,
        "tensor_split": tensor_split,
        "n_threads": max(1, requested_threads) if requested_threads > 0 else max(1, cores // 2),
        "n_threads_batch": max(1, cores),
        "n_batch": max(1, int(load_config.get("batchSize", 512))),
        "flash_attn": bool(load_config.get("flashAttention", False)),
        "use_mmap": bool(load_config.get("useMmap", True)),
        "use_mlock": bool(load_config.get("useMlock", False)),
        "verbose": True,
    }


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
                self._forwarder = RpcForwarder(peer, include_cpu=remote_cpu_layers > 0)
                rpc_endpoint = await self._forwarder.start()
            total_layers = remote_gpu_layers + remote_cpu_layers + local_layers
            if rpc_endpoint and total_layers <= 0:
                remote_gpu_layers = local_layers = 1
                total_layers = 2
            # Registers the worker RPC device and stages the exact device list,
            # so llama.cpp enumerates the worker ahead of the local GPU and
            # actually splits layers across both computers.
            tensor_split = await asyncio.to_thread(
                prepare_rpc_load, rpc_endpoint, remote_gpu_layers, remote_cpu_layers, local_layers
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
        self.store.log("INFO", "rpc_worker_ready", f"{endpoint} include_cpu={include_cpu}")
        host, port = endpoint.split(":", 1)
        return await asyncio.open_connection(host, int(port))

    async def chat(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str]
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
            return await asyncio.to_thread(self._chat_sync, messages, settings)

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
                                    loop.call_soon_threadsafe(
                                        queue.put_nowait, {"type": kind, "content": piece}
                                    )
                    for kind, piece in splitter.finish():
                        (reasoning_parts if kind == "reasoning" else answer_parts).append(piece)
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": kind, "content": piece})
                    full = "".join(reasoning_parts) + "".join(answer_parts)
                    completion_tokens = 1
                    with self._sync_lock:
                        completion_tokens = max(
                            1, len(self.llm.tokenize(full.encode("utf-8"), add_bos=False))
                        )
                    elapsed = max((time.monotonic() - started) if started else 0.001, 0.001)
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "stats", "tokensPerSecond": round(completion_tokens / elapsed, 2)},
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
                except BaseException as error:
                    loop.call_soon_threadsafe(queue.put_nowait, error)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            thread = threading.Thread(target=run, name="llama-chat-stream", daemon=True)
            thread.start()
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise BackendError("generation_failed", str(item)) from item
                yield item

    def _wire_messages(
        self, messages: list[dict[str, Any]], settings: dict[str, Any]
    ) -> list[dict[str, str]]:
        wire = [{"role": item["role"], "content": item.get("content", "")} for item in messages]
        system = settings.get("systemPrompt", "").strip()
        if system and not any(item["role"] == "system" for item in wire):
            wire.insert(0, {"role": "system", "content": system})
        return wire

    def _chat_sync(self, messages: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
        from llama_cpp import LogitsProcessorList

        wire = self._wire_messages(messages, settings)

        def abort_on_cancel(_tokens: object, logits: object) -> object:
            if self._cancel.is_set():
                raise InterruptedError
            return logits

        started = time.monotonic()
        with self._sync_lock:
            try:
                result = self.llm.create_chat_completion(
                    messages=wire,
                    temperature=float(settings.get("temperature", 0.7)),
                    max_tokens=int(settings.get("maxTokens", 512)),
                    logits_processor=LogitsProcessorList([abort_on_cancel]),
                )
            except InterruptedError:
                self.llm.reset()
                return {"content": "", "reasoning": "", "tokensPerSecond": 0.0}
        elapsed = max(time.monotonic() - started, 0.001)
        usage = result.get("usage") or {}
        completion = max(1, int(usage.get("completion_tokens") or 1))
        content = str(result["choices"][0]["message"].get("content") or "")
        reasoning, answer = split_reasoning(content, self.reasoning)
        return {
            "content": answer,
            "reasoning": reasoning,
            "tokensPerSecond": round(completion / elapsed, 2),
        }

    def cancel(self) -> None:
        self._cancel.set()

    async def benchmark(self) -> tuple[float, float]:
        async with self._async_lock:
            if not self.llm:
                raise BackendError("model_not_loaded", "Start the model before benchmarking it.")
            return await asyncio.to_thread(self._benchmark_sync)

    def _benchmark_sync(self) -> tuple[float, float]:
        prompt = "SharedLocalLLM benchmark: explain why local inference is useful in one paragraph."
        with self._sync_lock:
            tokens = self.llm.tokenize(prompt.encode(), add_bos=True)
            self.llm.reset()
            started = time.perf_counter()
            self.llm.eval(tokens)
            prompt_rate = len(tokens) / max(time.perf_counter() - started, 0.001)
            started = time.perf_counter()
            result = self.llm.create_completion(prompt=prompt, max_tokens=64, temperature=0.0)
            elapsed = max(time.perf_counter() - started, 0.001)
            text = str(result["choices"][0].get("text") or "")
            generated = max(1, len(self.llm.tokenize(text.encode(), add_bos=False)))
            return prompt_rate, generated / elapsed
