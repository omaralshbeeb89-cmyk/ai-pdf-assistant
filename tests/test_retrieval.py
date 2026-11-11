import numpy as np
import pytest

from pdf_assistant import retrieval
from pdf_assistant.library import build_collection

from conftest import EMBEDDING_DIM, make_document


CHUNKS = [
    "cats are small pets",           # 0
    "dogs are loyal pets",           # 1
    "linear regression model",       # 2
    "ridge regression penalty",      # 3
    "the lasso sets coefficients",   # 4
    "trees split the data",          # 5
    "boosting grows trees slowly",   # 6
    "support vector machines",       # 7
]


def query_vector(weights):
    """Fake query embedding: weights[i] = similarity to chunk i."""

    vector = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    for index, weight in weights.items():
        vector[0, index] = weight
    return vector / np.linalg.norm(vector)


@pytest.fixture
def collection():
    return build_collection([make_document("book.pdf", CHUNKS, pages=list(range(1, 9)))])


@pytest.fixture
def semantic(monkeypatch):
    """Set which chunks the query is semantically close to."""

    def set_weights(weights):
        monkeypatch.setattr(retrieval, "_embed_query", lambda query: query_vector(weights))

    return set_weights


def test_chunk_found_by_both_methods_ranks_first(collection, semantic):
    semantic({3: 0.9, 0: 0.4})

    results = retrieval.search(collection, "ridge penalty", top_k=3, rerank=False)

    assert results[0].chunk_index == 3
    assert results[0].faiss_rank == 1
    assert results[0].bm25_score > 0


def test_keyword_search_skips_non_matching_chunks(collection, semantic):
    # Semantic search finds chunks 0 and 1; only chunk 4 contains "lasso".
    semantic({0: 0.9, 1: 0.5})

    results = retrieval.search(
        collection, "lasso", top_k=5, candidate_k=2, rerank=False
    )

    # Chunks with a BM25 score of 0 must not get keyword credit.
    assert {result.chunk_index for result in results} == {0, 1, 4}

    keyword_only = next(result for result in results if result.chunk_index == 4)
    assert keyword_only.faiss_rank is None


def test_results_carry_page_source_and_text(collection, semantic):
    semantic({5: 1.0})

    result = retrieval.search(collection, "trees", top_k=1, rerank=False)[0]

    assert result.page == 6
    assert result.source == "book.pdf"
    assert result.text == CHUNKS[5]


def test_top_k_limits_results(collection, semantic):
    semantic({i: 1.0 / (i + 1) for i in range(len(CHUNKS))})

    assert len(retrieval.search(collection, "pets", top_k=2, rerank=False)) == 2


def test_query_without_words_uses_semantic_search_only(collection, semantic):
    semantic({7: 1.0})

    results = retrieval.search(collection, "???", top_k=1, rerank=False)

    assert results[0].chunk_index == 7
    assert results[0].bm25_score == 0


def test_rerank_receives_wider_pool(collection, semantic, monkeypatch):
    semantic({i: 1.0 / (i + 1) for i in range(len(CHUNKS))})
    received = {}

    def fake_rerank(query, results, top_k):
        received["pool"] = len(results)
        return results[::-1][:top_k]

    monkeypatch.setattr(retrieval, "rerank_results", fake_rerank)

    results = retrieval.search(collection, "pets", top_k=2, rerank=True)

    assert received["pool"] == len(CHUNKS)  # all 8, not just top_k
    assert len(results) == 2
