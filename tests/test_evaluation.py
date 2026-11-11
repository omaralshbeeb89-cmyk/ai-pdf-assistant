from pdf_assistant import evaluation, retrieval
from pdf_assistant.library import build_collection
from pdf_assistant.retrieval import SearchResult

from conftest import make_document


ITEM = {"question": "q", "page": 5, "chunk_index": 1}


def result(source, page, chunk_index=0):
    return SearchResult(
        chunk_index=chunk_index, source=source, page=page, text="",
        rrf_score=0.0, faiss_rank=None, faiss_score=None, bm25_score=0.0,
    )


def run_one(monkeypatch, results):
    monkeypatch.setattr(retrieval, "search", lambda *args, **kwargs: results)
    collection = build_collection([
        make_document("book.pdf", ["a", "b"]),
        make_document("other.pdf", ["c"]),
    ])
    return evaluation._evaluate_one(collection, "book.pdf", ITEM, rerank=False)


def test_hit_reports_rank_of_first_correct_page(monkeypatch):
    row = run_one(monkeypatch, [result("book.pdf", 9), result("book.pdf", 5, chunk_index=1)])

    assert row["page_rank"] == 2
    assert row["chunk_hit"]


def test_same_page_in_another_pdf_is_not_a_hit(monkeypatch):
    row = run_one(monkeypatch, [result("other.pdf", 5)])

    assert row["page_rank"] is None
    assert not row["chunk_hit"]


def test_metrics_are_averaged(monkeypatch):
    rows = iter([
        [result("book.pdf", 5)],          # rank 1
        [result("x.pdf", 1), result("book.pdf", 5)],  # rank 2
        [result("x.pdf", 1)],             # miss
        [result("book.pdf", 5)],          # rank 1
    ])
    monkeypatch.setattr(retrieval, "search", lambda *args, **kwargs: next(rows))
    monkeypatch.setattr(evaluation, "WORKERS", 1)  # keep the order fixed

    collection = build_collection([make_document("book.pdf", ["a", "b"])])
    report = evaluation.evaluate(collection, [ITEM] * 4, rerank=False)

    assert report["page_hit"] == 0.75
    assert report["page_mrr"] == (1 + 0.5 + 0 + 1) / 4
    assert len(report["misses"]) == 1
