from pdf_assistant.rerank import rerank
from pdf_assistant.retrieval import SearchResult


def candidates(count):
    return [
        SearchResult(
            chunk_index=i, source="a.pdf", page=i + 1, text=f"passage {i}",
            rrf_score=1.0 / (i + 1), faiss_rank=i + 1, faiss_score=0.5, bm25_score=0.0,
        )
        for i in range(count)
    ]


def ids(results):
    return [result.chunk_index for result in results]


def test_orders_by_model_score(fake_client):
    fake_client.output_text = {"scores": [
        {"id": 0, "score": 2}, {"id": 1, "score": 9}, {"id": 2, "score": 5},
    ]}

    results = rerank("query", candidates(3), top_k=2)

    assert ids(results) == [1, 2]
    assert [result.rerank_score for result in results] == [9, 5]


def test_ties_keep_hybrid_order(fake_client):
    fake_client.output_text = {"scores": [
        {"id": 0, "score": 7}, {"id": 1, "score": 7}, {"id": 2, "score": 7},
    ]}

    assert ids(rerank("query", candidates(3), top_k=3)) == [0, 1, 2]


def test_passages_missing_from_reply_go_last(fake_client):
    fake_client.output_text = {"scores": [{"id": 2, "score": 1}]}

    assert ids(rerank("query", candidates(3), top_k=3)) == [2, 0, 1]


def test_unknown_ids_are_ignored(fake_client):
    fake_client.output_text = {"scores": [{"id": 99, "score": 10}, {"id": 1, "score": 3}]}

    assert ids(rerank("query", candidates(2), top_k=2)) == [1, 0]


def test_api_failure_falls_back_to_hybrid_order(fake_client):
    fake_client.error = RuntimeError("rate limited")

    results = rerank("query", candidates(4), top_k=2)

    assert ids(results) == [0, 1]
    assert all(result.rerank_score is None for result in results)


def test_invalid_json_falls_back_to_hybrid_order(fake_client):
    fake_client.output_text = "not json"

    assert ids(rerank("query", candidates(3), top_k=2)) == [0, 1]


def test_prompt_contains_query_and_every_passage(fake_client):
    fake_client.output_text = {"scores": []}

    rerank("what is ridge?", candidates(3), top_k=3)

    prompt = fake_client.calls[0]["input"]
    assert "what is ridge?" in prompt
    assert all(f"passage {i}" in prompt for i in range(3))


def test_empty_input_makes_no_api_call():
    # The autouse block_openai fixture fails the test on any API call.
    assert rerank("query", [], top_k=5) == []
