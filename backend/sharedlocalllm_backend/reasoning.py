from __future__ import annotations

import re

_NAME = re.compile(r"(r1|qwq|deepseek|qwen3|reasoner|reasoning|thinking)", re.IGNORECASE)

# End markers emitted by common reasoning models. The ``...`` marker is only
# trusted for models whose reasoning section starts with it, so plain text
# ellipses are never treated as a reasoning boundary.
_SPECIFIC_ENDS = ("</reasoning>", "<|end_of_reasoning|>", "<|end_of_thought|>", " Watermark")
_ELLIPSIS = "..."
_LOOKAHEAD = 24

_START = re.compile(r"^\s*(?:<reasoning>|<\|begin_of_thought\|>|<\|reasoning_content\|>|\.\.\.)\s*")
_START_MARKERS = ("<reasoning>", "<|begin_of_thought|>", "<|reasoning_content|>")
_ANSWER_PREFIX = re.compile(r"^\s*Assistant:\s*<\|reserved_actor\|>\s*")
_DOT_RUN = re.compile(r"\.{3,}")


def is_reasoning_model(name: str) -> bool:
    return bool(_NAME.search(name))


def _clean(text: str) -> str:
    return _START.sub("", text).strip()


def _clean_answer(text: str) -> str:
    return _ANSWER_PREFIX.sub("", text).strip()


def split_reasoning(text: str, reasoning: bool) -> tuple[str, str]:
    """Split a finished model response into (reasoning, answer)."""
    body = text.strip()
    if not body:
        return "", body
    start = body.find("<reasoning>")
    if start >= 0:
        end = body.find("</reasoning>", start + len("<reasoning>"))
        if end >= 0:
            return _clean(body[start + len("<reasoning>"):end]), _clean_answer(body[end + len("</reasoning>"):])
    if not reasoning:
        return "", body
    if body.startswith(_ELLIPSIS):
        match = _DOT_RUN.search(body, len(_ELLIPSIS))
        if match:
            reasoning_end = match.end() - 3
            return _clean(body[len(_ELLIPSIS):reasoning_end]), _clean_answer(body[match.end():])
    for marker in _SPECIFIC_ENDS:
        end = body.find(marker)
        if end >= 0:
            return _clean(body[:end]), _clean_answer(body[end + len(marker):])
    return "", body


class ReasoningStreamSplitter:
    """Route streamed tokens to reasoning or answer as they arrive.

    Reasoning models are detected up front from the model name. Their output
    is treated as reasoning until an end marker closes the section, after
    which everything is the answer. Models not detected as reasoning stream
    straight through as answer.
    """

    def __init__(self, reasoning: bool) -> None:
        self._reasoning = reasoning
        self._phase = "pending"
        self._buffer = ""
        self._ellipsis = False

    def push(self, text: str) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        self._buffer += text
        if self._phase == "content":
            events.append(("token", text))
            self._buffer = ""
            return events
        if self._phase == "pending":
            starts = [
                (self._buffer.find(value), value)
                for value in _START_MARKERS
                if self._buffer.find(value) >= 0
            ]
            if starts:
                marker, start_marker = min(starts, key=lambda value: value[0])
                prefix = self._buffer[:marker]
                if prefix:
                    events.append(("token", prefix))
                self._buffer = self._buffer[marker + len(start_marker):]
                self._phase = "reasoning"
            elif self._reasoning:
                stripped = self._buffer.lstrip()
                if stripped.startswith(_ELLIPSIS):
                    self._ellipsis = True
                    self._buffer = stripped[len(_ELLIPSIS):]
                    self._phase = "reasoning"
                else:
                    found = self._find_specific_end()
                    if found is not None:
                        index, length = found
                        events.append(("reasoning", _clean(self._buffer[:index])))
                        tail = self._buffer[index + length:]
                        if tail:
                            events.append(("token", tail))
                        self._buffer = ""
                        self._phase = "content"
                    return events
            else:
                safe = max(0, len(self._buffer) - _LOOKAHEAD)
                if safe:
                    events.append(("token", self._buffer[:safe]))
                    self._buffer = self._buffer[safe:]
                return events
        found = self._find_end()
        if found is not None:
            index, length = found
            tail = index + length
            if self._ellipsis:
                while tail < len(self._buffer) and self._buffer[tail] == ".":
                    tail += 1
                reasoning_end = tail - 3
            else:
                reasoning_end = index
            events.append(("reasoning", _clean(self._buffer[:reasoning_end])))
            if tail < len(self._buffer):
                events.append(("token", self._buffer[tail:]))
            self._buffer = ""
            self._phase = "content"
            return events
        safe = max(0, len(self._buffer) - _LOOKAHEAD)
        if safe:
            events.append(("reasoning", _clean(self._buffer[:safe])))
            self._buffer = self._buffer[safe:]
        return events

    def finish(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        if self._buffer:
            if self._phase == "reasoning" and self._ellipsis:
                events.append(("reasoning", _clean(self._buffer)))
            else:
                events.append(("token", self._buffer))
            self._buffer = ""
        return events

    def _find_end(self) -> tuple[int, int] | None:
        candidates = [(self._buffer.find(marker), len(marker)) for marker in _SPECIFIC_ENDS]
        if self._ellipsis:
            index = self._buffer.find(_ELLIPSIS)
            if index >= 0:
                candidates.append((index, len(_ELLIPSIS)))
        candidates = [candidate for candidate in candidates if candidate[0] >= 0]
        return min(candidates, key=lambda candidate: candidate[0]) if candidates else None

    def _find_specific_end(self) -> tuple[int, int] | None:
        candidates = [
            (self._buffer.find(marker), len(marker)) for marker in _SPECIFIC_ENDS
        ]
        candidates = [candidate for candidate in candidates if candidate[0] >= 0]
        return min(candidates, key=lambda candidate: candidate[0]) if candidates else None
