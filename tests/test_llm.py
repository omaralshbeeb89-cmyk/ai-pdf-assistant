from types import SimpleNamespace

import pytest

from pdf_assistant import config, llm
from pdf_assistant.retrieval import SearchResult


HISTORY = [
    {"role": "user", "content": "What is the lasso?"},
    {"role": "assistant", "content": "A shrinkage method."},
]


def result(source, page, text):
    return SearchResult(
        chunk_index=0, source=source, page=page, text=text,
        rrf_score=0.0, faiss_rank=None, faiss_score=None, bm25_score=0.0,
    )


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


# ---------- query rewriting ----------

def test_first_question_is_not_rewritten():
    # No history, so no API call (block_openai would fail the test).
    assert llm.rewrite_question("What is OLS?", []) == "What is OLS?"


def test_follow_up_is_rewritten_with_history(fake_client):
    fake_client.output_text = "  Why is the lasso useful?  "

    assert llm.rewrite_question("why is it useful?", HISTORY) == "Why is the lasso useful?"
    assert "What is the lasso?" in fake_client.calls[0]["input"]


def test_rewrite_failure_keeps_original_question(fake_client):
    fake_client.error = RuntimeError("timeout")

    assert llm.rewrite_question("why?", HISTORY) == "why?"


def test_empty_rewrite_keeps_original_question(fake_client):
    fake_client.output_text = "   "

    assert llm.rewrite_question("why?", HISTORY) == "why?"


# ---------- history ----------

def test_trim_history_keeps_latest_messages():
    history = [{"role": "user", "content": str(i)} for i in range(20)]

    trimmed = llm.trim_history(history)

    assert len(trimmed) == config.MAX_HISTORY_MESSAGES
    assert trimmed[-1]["content"] == "19"


def test_format_empty_history():
    assert llm.format_history([]) == "No previous conversation."


# ---------- answer prompt ----------

def test_single_pdf_prompt_uses_page_citations():
    prompt = llm.build_answer_prompt(
        "What is ridge?", [result("a.pdf", 12, "Ridge shrinks coefficients.")], []
    )

    assert "[PDF page 12]\nRidge shrinks coefficients." in prompt
    assert "Cite factual claims using [PDF page N]" in prompt
    assert "a.pdf" not in prompt


def test_multi_pdf_prompt_names_the_file():
    prompt = llm.build_answer_prompt(
        "What is ridge?", [result("a.pdf", 12, "Ridge shrinks.")], [], multi_doc=True
    )

    assert "[a.pdf, page 12]\nRidge shrinks." in prompt
    assert "[file name, page N]" in prompt


def test_prompt_includes_history_and_question():
    prompt = llm.build_answer_prompt("why?", [result("a.pdf", 1, "x")], HISTORY)

    assert "USER: What is the lasso?" in prompt
    assert "ASSISTANT: A shrinkage method." in prompt
    assert prompt.rstrip().endswith("why?\n\nANSWER:")


# ---------- streaming ----------

def test_stream_yields_text_pieces(fake_client):
    fake_client.stream_events = [
        SimpleNamespace(type="response.created"),
        delta("Ridge "),
        delta("shrinks."),
        SimpleNamespace(type="response.completed"),
    ]

    pieces = list(llm.stream_answer("q", [result("a.pdf", 1, "x")], []))

    assert pieces == ["Ridge ", "shrinks."]
    assert fake_client.calls[0]["stream"] is True


def test_stream_raises_on_failed_response(fake_client):
    fake_client.stream_events = [
        delta("Partial"),
        SimpleNamespace(type="response.failed", response=SimpleNamespace(error="server error")),
    ]

    with pytest.raises(RuntimeError, match="server error"):
        list(llm.stream_answer("q", [result("a.pdf", 1, "x")], []))
