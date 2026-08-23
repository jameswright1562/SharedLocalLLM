from __future__ import annotations

import ctypes
import sys
from typing import Any

# ggml_log_level values from ggml.h. Their numeric order is not severity order.
GGML_LOG_LEVEL_NONE = 0
GGML_LOG_LEVEL_INFO = 1
GGML_LOG_LEVEL_WARN = 2
GGML_LOG_LEVEL_ERROR = 3
GGML_LOG_LEVEL_DEBUG = 4
GGML_LOG_LEVEL_CONT = 5

_VISIBLE_LEVELS = frozenset({
    GGML_LOG_LEVEL_INFO, GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_ERROR,
})
_callback_ref: Any | None = None
_show_continuations = True


def filtered_native_log(
    level: int, text: bytes, _user_data: object | None = None
) -> None:
    """Keep performance reports and errors while dropping loader noise."""
    global _show_continuations
    decoded = text.decode("utf-8", "replace")
    if level != GGML_LOG_LEVEL_CONT:
        unused_tensor = level == GGML_LOG_LEVEL_WARN and (
            "model has unused tensor " in decoded and " -- ignoring" in decoded
        )
        _show_continuations = level in _VISIBLE_LEVELS and not unused_tensor
    if _show_continuations:
        sys.stderr.write(decoded)
        sys.stderr.flush()


def install_native_log_filter() -> bool:
    """Replace llama-cpp-python's default native callback exactly once."""
    global _callback_ref
    if _callback_ref is not None:
        return False
    try:
        from llama_cpp import llama_cpp as low
    except ImportError:
        return False
    callback = low.llama_log_callback(filtered_native_log)
    low.llama_log_set(callback, ctypes.c_void_p(0))
    # ctypes callbacks must remain reachable while native code may invoke them.
    _callback_ref = callback
    return True
