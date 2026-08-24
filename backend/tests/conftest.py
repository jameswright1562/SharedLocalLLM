"""Shared pytest configuration for the backend unit tests.

The lightweight CI environment intentionally does not install the CUDA-heavy
``llama-cpp-python`` wheel. Unit tests never load a real model, so when the
binding is absent we register a minimal stub with the exact surface the lazy
imports touch (Llama, LogitsProcessorList, context default params). Tests that
need real native bindings already guard themselves with ``importorskip``.
"""

from __future__ import annotations

import importlib.util
import sys
import types


def _install_llama_cpp_stub() -> None:
    if importlib.util.find_spec("llama_cpp") is not None:
        return

    lowlevel = types.ModuleType("llama_cpp.llama_cpp")

    def llama_context_default_params() -> types.SimpleNamespace:
        return types.SimpleNamespace(kv_unified=False, n_ctx=512, n_batch=2048, n_ubatch=512)

    lowlevel.llama_context_default_params = llama_context_default_params  # type: ignore[attr-defined]

    package = types.ModuleType("llama_cpp")
    package.__version__ = "0.0.0-stub"
    package.__file__ = __file__
    # Lets tests that need the real native registry distinguish the stub.
    package.__llama_stub__ = True
    package.Llama = type("Llama", (), {})
    package.LogitsProcessorList = list
    package.llama_cpp = lowlevel

    sys.modules["llama_cpp"] = package
    sys.modules["llama_cpp.llama_cpp"] = lowlevel


_install_llama_cpp_stub()
