from types import SimpleNamespace

import pytest

from pdf_assistant import config, llm, usage
from pdf_assistant.retrieval import SearchResult


@pytest.fixture(autouse=True)
def fixed_prices(monkeypatch):
    monkeypatch.setattr(config, "PRICES_PER_MILLION", {
        "chat": (1.0, 4.0),
        "embed": (0.1, 0.0),
    })


def test_cost_uses_input_and_output_prices():
    assert usage.cost_of("chat", 1_000_000, 500_000) == pytest.approx(1.0 + 2.0)


def test_unknown_model_costs_nothing():
    assert usage.cost_of("mystery-model", 1_000_000, 1_000_000) == 0


def test_calls_inside_track_are_recorded():
    with usage.track() as tracked:
        usage.record("rewrite", "chat", 1000, 100)
        usage.record("search", "embed", 10)

    data = tracked.to_dict()

    assert data["input_tokens"] == 1010
    assert data["output_tokens"] == 100
    assert data["cost"] == pytest.approx((1000 * 1.0 + 100 * 4.0 + 10 * 0.1) / 1e6)
    assert [call["purpose"] for call in data["calls"]] == ["rewrite", "search"]


def test_recording_outside_track_does_nothing():
    usage.record("rewrite", "chat", 1000, 100)  # must not raise

    with usage.track() as tracked:
        pass

    assert tracked.calls == []


def test_nested_track_blocks_are_separate():
    with usage.track() as outer:
        usage.record("a", "chat", 1)

        with usage.track() as inner:
            usage.record("b", "chat", 2)

        usage.record("c", "chat", 3)

    assert [call.purpose for call in outer.calls] == ["a", "c"]
    assert [call.purpose for call in inner.calls] == ["b"]


def test_record_response_reads_responses_api_usage():
    with usage.track() as tracked:
        usage.record_response("answer", "chat", SimpleNamespace(input_tokens=50, output_tokens=7))
        usage.record_response("answer", "chat", None)  # no usage reported

    assert (tracked.calls[0].input_tokens, tracked.calls[0].output_tokens) == (50, 7)
    assert len(tracked.calls) == 1


def test_streamed_answer_records_usage_from_completed_event(fake_client, monkeypatch):
    monkeypatch.setattr(config, "CHAT_MODEL", "chat")
    fake_client.stream_events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hi"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=SimpleNamespace(input_tokens=900, output_tokens=40)),
        ),
    ]
    passage = SearchResult(0, "a.pdf", 1, "text", 0.0, None, None, 0.0)

    with usage.track() as tracked:
        assert list(llm.stream_answer("q", [passage], [])) == ["Hi"]

    assert tracked.calls[0].purpose == "answer"
    assert tracked.calls[0].input_tokens == 900
    assert tracked.calls[0].output_tokens == 40


@pytest.mark.parametrize("cost, text", [
    (0, "$0"),
    (0.00004, "<$0.0001"),
    (0.00123, "~$0.0012"),
])
def test_format_cost(cost, text):
    assert usage.format_cost(cost) == text


def test_format_usage_and_total():
    data = usage.total([
        {"input_tokens": 1000, "output_tokens": 200, "cost": 0.001},
        None,
        {"input_tokens": 500, "output_tokens": 34, "cost": 0.0005},
    ])

    assert usage.format_usage(data) == "1,734 tokens · ~$0.0015"
