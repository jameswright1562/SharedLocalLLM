from __future__ import annotations

import sys
import types

import pytest

from sharedlocalllm_backend import native_logging


@pytest.fixture(autouse=True)
def reset_log_filter_state():
    native_logging._callback_ref = None
    native_logging._show_continuations = True
    yield
    native_logging._callback_ref = None
    native_logging._show_continuations = True


def test_info_and_error_lines_are_kept(capsys: pytest.CaptureFixture[str]) -> None:
    native_logging.filtered_native_log(
        native_logging.GGML_LOG_LEVEL_INFO, b"prompt eval time =   512.0 ms\n"
    )
    native_logging.filtered_native_log(
        native_logging.GGML_LOG_LEVEL_ERROR, b"failed to load model\n"
    )
    captured = capsys.readouterr()
    assert "prompt eval time" in captured.err
    assert "failed to load model" in captured.err


def test_debug_and_warn_noise_is_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    native_logging.filtered_native_log(
        native_logging.GGML_LOG_LEVEL_DEBUG,
        b"create_tensor: loading tensor blk.64.nextn.eh_proj.weight\n",
    )
    native_logging.filtered_native_log(
        native_logging.GGML_LOG_LEVEL_WARN,
        b"model has unused tensor blk.64.ffn_up.weight (size = 73113600 bytes) -- ignoring\n",
    )
    assert capsys.readouterr().err == ""


def test_actionable_warnings_are_kept(capsys: pytest.CaptureFixture[str]) -> None:
    native_logging.filtered_native_log(
        native_logging.GGML_LOG_LEVEL_WARN, b"context size exceeds the trained limit\n"
    )
    assert "context size exceeds" in capsys.readouterr().err


def test_none_level_is_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    native_logging.filtered_native_log(native_logging.GGML_LOG_LEVEL_NONE, b"hidden\n")
    assert capsys.readouterr().err == ""


def test_continuation_lines_follow_the_previous_decision(
    capsys: pytest.CaptureFixture[str],
) -> None:
    native_logging.filtered_native_log(
        native_logging.GGML_LOG_LEVEL_INFO, b"eval time =     120.0 ms\n"
    )
    native_logging.filtered_native_log(native_logging.GGML_LOG_LEVEL_CONT, b" (kept)\n")
    native_logging.filtered_native_log(
        native_logging.GGML_LOG_LEVEL_DEBUG, b"create_tensor: loading tensor blk.1\n"
    )
    native_logging.filtered_native_log(native_logging.GGML_LOG_LEVEL_CONT, b" (hidden)\n")
    captured = capsys.readouterr()
    assert "(kept)" in captured.err
    assert "(hidden)" not in captured.err


def test_install_registers_the_callback_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[tuple[object, object]] = []

    class FakeCallbackType:
        def __init__(self, function: object) -> None:
            self.function = function

    stub = types.SimpleNamespace(
        llama_log_callback=FakeCallbackType,
        llama_log_set=lambda callback, user_data: registered.append((callback, user_data)),
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(llama_cpp=stub))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", stub)

    assert native_logging.install_native_log_filter() is True
    assert len(registered) == 1
    callback, _user_data = registered[0]
    assert isinstance(callback, FakeCallbackType)
    assert callback.function is native_logging.filtered_native_log

    assert native_logging.install_native_log_filter() is False
    assert len(registered) == 1


def test_install_survives_a_missing_llama_cpp_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    assert native_logging.install_native_log_filter() is False
    assert native_logging._callback_ref is None
