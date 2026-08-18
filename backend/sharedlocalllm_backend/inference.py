from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
from typing import Any, AsyncIterator

from .errors import BackendError
from .peer import RpcForwarder
from .rpc_native import NativeRpcServer


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

    async def load(
        self, model: dict[str, Any], path: str, load_config: dict[str, Any],
        peer: Any, local_id: str, peer_id: str | None,
    ) -> None:
        async with self._async_lock:
            await self._unload_locked()
            allocations = load_config.get("gpuLayers") or []
            rpc_endpoint = None
            remote_layers = 0
            local_layers = 0
            for allocation in allocations:
                if peer_id and allocation.get("nodeId") == peer_id:
                    remote_layers += int(allocation.get("layers", 0))
                elif allocation.get("nodeId") == local_id:
                    local_layers += int(allocation.get("layers", 0))
            if peer_id and (remote_layers > 0 or model.get("fit") == "combined-gpu"):
                self._forwarder = RpcForwarder(peer)
                rpc_endpoint = await self._forwarder.start()
            total_layers = remote_layers + local_layers
            tensor_split = None
            if rpc_endpoint:
                if total_layers <= 0:
                    remote_layers = local_layers = 1
                    total_layers = 2
                # llama.cpp places registered RPC devices before local GPUs.
                tensor_split = [float(remote_layers), float(local_layers)]
            context = max(512, int(load_config.get("contextSize", 4096)))
            self.store.log(
                "INFO", "model_load",
                f"model={model['name']} ctx={context} rpc={rpc_endpoint} split={tensor_split}",
            )
            try:
                await asyncio.to_thread(
                    self._load_sync, path, context, total_layers, tensor_split, rpc_endpoint
                )
            except Exception as error:
                await self._stop_forwarder()
                raise BackendError(
                    "model_load_failed", str(error),
                    "Reduce context/model size or review llama-cpp-python CUDA/RPC logs."
                ) from error
            self.model_id = model["id"]
            self.store.log("INFO", "model_loaded", model["name"])

    def _load_sync(
        self, path: str, context: int, gpu_layers: int,
        tensor_split: list[float] | None, rpc_endpoint: str | None,
    ) -> None:
        from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER

        with self._sync_lock:
            self.llm = Llama(
                model_path=path,
                n_ctx=context,
                n_gpu_layers=gpu_layers if gpu_layers > 0 else -1,
                split_mode=LLAMA_SPLIT_MODE_LAYER,
                tensor_split=tensor_split,
                rpc_servers=rpc_endpoint,
                n_threads=max(1, (os.cpu_count() or 8) // 2),
                n_threads_batch=max(1, os.cpu_count() or 8),
                verbose=True,
            )

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

    async def open_rpc_worker_connection(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        endpoint = await asyncio.to_thread(self._rpc_worker.start)
        self.store.log("INFO", "rpc_worker_ready", endpoint)
        host, port = endpoint.split(":", 1)
        return await asyncio.open_connection(host, int(port))

    async def chat(
        self, messages: list[dict[str, Any]], settings: dict[str, Any], images: list[str]
    ) -> str:
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
        self, messages: list[dict[str, Any]], settings: dict[str, Any]
    ) -> AsyncIterator[str]:
        async with self._async_lock:
            if not self.llm:
                raise BackendError("model_not_loaded", "Start a model before chatting.")
            self._cancel.clear()
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()

            def run() -> None:
                try:
                    wire = self._wire_messages(messages, settings)
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
                                loop.call_soon_threadsafe(queue.put_nowait, str(content))
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

    def _chat_sync(self, messages: list[dict[str, Any]], settings: dict[str, Any]) -> str:
        from llama_cpp import StoppingCriteriaList

        wire = self._wire_messages(messages, settings)
        criteria = StoppingCriteriaList([lambda _tokens, _logits: self._cancel.is_set()])
        with self._sync_lock:
            result = self.llm.create_chat_completion(
                messages=wire,
                temperature=float(settings.get("temperature", 0.7)),
                max_tokens=int(settings.get("maxTokens", 512)),
                stopping_criteria=criteria,
            )
        return str(result["choices"][0]["message"].get("content") or "")

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
