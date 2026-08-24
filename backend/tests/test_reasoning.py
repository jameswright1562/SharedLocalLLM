from __future__ import annotations

from sharedlocalllm_backend.reasoning import ReasoningStreamSplitter, is_reasoning_model, split_reasoning


def drain(pieces: list[str], reasoning: bool) -> tuple[list[str], list[str]]:
    splitter = ReasoningStreamSplitter(reasoning)
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    for piece in pieces:
        for kind, content in splitter.push(piece):
            (reasoning_parts if kind == "reasoning" else answer_parts).append(content)
    for kind, content in splitter.finish():
        (reasoning_parts if kind == "reasoning" else answer_parts).append(content)
    return reasoning_parts, answer_parts


def test_reasoning_name_detection() -> None:
    assert is_reasoning_model("DeepSeek-R1-Distill-Qwen-7B")
    assert is_reasoning_model("QwQ-32B")
    assert is_reasoning_model("Qwen3-14B")
    assert not is_reasoning_model("Llama-3.1-8B-Instruct")


def test_split_reasoning_without_reasoning_flag_returns_answer_only() -> None:
    text = "I think step by step... but the final answer is 42."
    assert split_reasoning(text, reasoning=False) == ("", text)


def test_split_reasoning_plain_response_keeps_answer() -> None:
    text = "The answer is 42."
    assert split_reasoning(text, reasoning=True) == ("", text)


def test_split_reasoning_deepseek_watermark_style() -> None:
    reason = "Let me multiply 6 by 7."
    text = f"{reason} Watermark\n\nThe answer is 42."
    assert split_reasoning(text, reasoning=True) == (reason, "The answer is 42.")


def test_split_reasoning_ellipsis_style() -> None:
    reason = "I should add the two numbers."
    text = f"...{reason}...The total is 7."
    assert split_reasoning(text, reasoning=True) == (reason, "The total is 7.")


def test_split_reasoning_thought_tags() -> None:
    reason = "Reason about the prompt."
    text = f"<|begin_of_thought|>{reason}<|end_of_thought|>Done."
    assert split_reasoning(text, reasoning=True) == (reason, "Done.")


def test_split_reasoning_reasoning_tags() -> None:
    reason = "Reason about the prompt."
    text = f"<reasoning>{reason}</reasoning>Done."
    assert split_reasoning(text, reasoning=False) == (reason, "Done.")


def test_split_reasoning_qwen_think_tags() -> None:
    reason = "Multiply 6 by 7."
    text = f"<think>\n{reason}\n</think>The answer is 42."
    assert split_reasoning(text, reasoning=True) == (reason, "The answer is 42.")


def test_stream_qwen_think_tags_split_across_chunks() -> None:
    reasoning_parts, answer_parts = drain(
        ["<th", "ink>Let me check.\n", "6x7=42.", "</th", "ink>Done."], reasoning=True
    )
    assert "".join(reasoning_parts) == "Let me check.\n6x7=42."
    assert "".join(answer_parts) == "Done."


def test_split_reasoning_cleans_reserved_actor_residue() -> None:
    reason = "A short reason."
    text = f"<|reasoning_content|>{reason}<|end_of_reasoning|>Assistant: <|reserved_actor|>Answer."
    assert split_reasoning(text, reasoning=True) == (reason, "Answer.")


def test_stream_non_reasoning_model_passes_through() -> None:
    reasoning_parts, answer_parts = drain(["Hello", " world"], reasoning=False)
    assert reasoning_parts == []
    assert "".join(answer_parts) == "Hello world"


def test_stream_reasoning_model_splits_across_chunks() -> None:
    reasoning_parts, answer_parts = drain(
        ["Let me", " check the math.", " Watermark", "Final"], reasoning=True
    )
    assert "".join(reasoning_parts) == "Let me check the math."
    assert "".join(answer_parts) == "Final"


def test_stream_ellipsis_model_splits() -> None:
    reasoning_parts, answer_parts = drain(["...", "Think ", "hard.", " ...", "The answer."], reasoning=True)
    assert "".join(reasoning_parts) == "Think hard."
    assert "".join(answer_parts) == "The answer."


def test_stream_flagged_model_without_markers_falls_back_to_answer() -> None:
    reasoning_parts, answer_parts = drain(["No ", "reasoning ", "here."], reasoning=True)
    assert reasoning_parts == []
    assert "".join(answer_parts) == "No reasoning here."


def test_stream_long_unmarked_answer_is_not_hidden_as_reasoning() -> None:
    text = "This is a perfectly ordinary long answer with no hidden reasoning markers at all."
    reasoning_parts, answer_parts = drain([text[:30], text[30:]], reasoning=True)
    assert reasoning_parts == []
    assert "".join(answer_parts) == text
