import numpy as np
import pytest

from pdf_assistant import retrieval
from pdf_assistant.library import build_collection, format_sources
from pdf_assistant.retrieval import SearchResult

from conftest import make_document


def result(source, page):
    return SearchResult(
        chunk_index=0, source=source, page=page, text="",
        rrf_score=0.0, faiss_rank=None, faiss_score=None, bm25_score=0.0,
    )


def test_single_document_reuses_its_indexes():
    doc = make_document("a.pdf", ["one", "two"])

    collection = build_collection([doc])

    assert collection.index is doc.index
    assert collection.bm25 is doc.bm25
    assert collection.chunk_sources == ["a.pdf", "a.pdf"]
    assert not collection.is_multi


def test_several_documents_are_combined_in_order():
    a = make_document("a.pdf", ["alpha one", "alpha two"], pages=[1, 2])
    b = make_document("b.pdf", ["beta one"], pages=[7])

    collection = build_collection([a, b])

    assert collection.is_multi
    assert collection.names == ["a.pdf", "b.pdf"]
    assert collection.chunks == ["alpha one", "alpha two", "beta one"]
    assert collection.chunk_pages == [1, 2, 7]
    assert collection.chunk_sources == ["a.pdf", "a.pdf", "b.pdf"]
    assert collection.index.ntotal == 3
    assert collection.page_count == 2 + 7


def test_same_file_selected_twice_is_used_once():
    first = make_document("a.pdf", ["text"], pdf_hash="same")
    copy = make_document("uploads-a.pdf", ["text"], pdf_hash="same")

    collection = build_collection([first, copy])

    assert not collection.is_multi
    assert collection.names == ["a.pdf"]


def test_empty_selection_is_an_error():
    with pytest.raises(ValueError):
        build_collection([])


def test_search_across_documents_finds_the_right_file(monkeypatch):
    a = make_document("stats.pdf", ["ridge regression", "lasso"], embeddings=np.eye(4)[:2])
    b = make_document("network.pdf", ["ospf routing", "bgp peers"], embeddings=np.eye(4)[2:])
    collection = build_collection([a, b])

    # Query vector closest to b's first chunk (global index 2).
    monkeypatch.setattr(
        retrieval, "_embed_query",
        lambda query: np.eye(4, dtype=np.float32)[2:3],
    )

    top = retrieval.search(collection, "ospf", top_k=1, rerank=False)[0]

    assert (top.source, top.text) == ("network.pdf", "ospf routing")


def test_format_sources_single_document():
    results = [result("a.pdf", 12), result("a.pdf", 3), result("a.pdf", 12)]

    assert format_sources(results, multi_doc=False) == "3, 12"


def test_format_sources_groups_by_file():
    results = [result("a.pdf", 12), result("b.pdf", 5), result("a.pdf", 3)]

    assert format_sources(results, multi_doc=True) == "a.pdf p. 3, 12; b.pdf p. 5"
