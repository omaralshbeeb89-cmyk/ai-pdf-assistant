import logging

import numpy as np
import pytest

from pdf_assistant import config, ingest

from conftest import make_pdf


LONG_PAGE = " ".join(f"word{i}" for i in range(500))  # ~3900 characters


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    path = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", path)
    return path


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace the embedding API with deterministic vectors; count calls."""

    calls = []

    def embed(chunks, progress=None):
        calls.append(len(chunks))
        rng = np.random.default_rng(len(chunks))
        return rng.random((len(chunks), 8), dtype=np.float32)

    monkeypatch.setattr(ingest, "_embed", embed)
    return calls


def test_load_document_builds_chunks_and_indexes(tmp_path, cache_dir, fake_embed):
    pdf = make_pdf(tmp_path / "book.pdf", [LONG_PAGE, "", "A short last page"])

    doc = ingest.load_document(pdf)

    assert doc.name == "book.pdf"
    assert doc.page_count == 3
    assert len(doc.chunks) > 2
    assert doc.chunks[-1] == "A short last page"
    # The blank page 2 produces no chunks.
    assert set(doc.chunk_pages) == {1, 3}
    assert doc.index.ntotal == len(doc.chunks)
    assert fake_embed == [len(doc.chunks)]


def test_second_load_uses_cache(tmp_path, cache_dir, fake_embed):
    pdf = make_pdf(tmp_path / "book.pdf", [LONG_PAGE])

    first = ingest.load_document(pdf)
    second = ingest.load_document(pdf)

    assert len(fake_embed) == 1, "embeddings were created twice"
    assert second.chunks == first.chunks
    np.testing.assert_array_equal(second.embeddings, first.embeddings)
    assert (cache_dir / "book_embeddings.npz").exists()
    assert (cache_dir / "book_faiss_index.bin").exists()


def test_changed_pdf_rebuilds_cache(tmp_path, cache_dir, fake_embed):
    pdf = tmp_path / "book.pdf"

    make_pdf(pdf, [LONG_PAGE])
    ingest.load_document(pdf)

    make_pdf(pdf, ["Completely different content"])
    doc = ingest.load_document(pdf)

    assert len(fake_embed) == 2
    assert doc.chunks == ["Completely different content"]


def test_corrupted_cache_is_rebuilt(tmp_path, cache_dir, fake_embed):
    pdf = make_pdf(tmp_path / "book.pdf", [LONG_PAGE])
    ingest.load_document(pdf)

    (cache_dir / "book_embeddings.npz").write_bytes(b"not a zip file")

    doc = ingest.load_document(pdf)

    assert len(fake_embed) == 2
    assert len(doc.chunks) > 1


def test_warns_about_pages_without_text(tmp_path, cache_dir, fake_embed, caplog):
    pdf = make_pdf(tmp_path / "book.pdf", ["Some text", "", ""])

    with caplog.at_level(logging.WARNING):
        ingest.load_document(pdf)

    assert "2 of 3 pages had no extractable text" in caplog.text


def test_pdf_without_any_text_is_an_error(tmp_path, cache_dir, fake_embed):
    pdf = make_pdf(tmp_path / "scan.pdf", ["", ""])

    with pytest.raises(ValueError, match="No extractable text"):
        ingest.load_document(pdf)


def test_missing_pdf_is_an_error(tmp_path, cache_dir):
    with pytest.raises(FileNotFoundError):
        ingest.load_document(tmp_path / "missing.pdf")
