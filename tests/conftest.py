"""Shared test helpers. No test may call the real OpenAI API."""

import json
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from pdf_assistant import config
from pdf_assistant.ingest import Document, tokenize


EMBEDDING_DIM = 16


@pytest.fixture(autouse=True)
def block_openai(monkeypatch):
    """Fail loudly if code tries to reach OpenAI without a fake client."""

    def no_network():
        raise AssertionError("Test tried to call the real OpenAI API.")

    monkeypatch.setattr(config, "get_client", no_network)


class FakeClient:
    """
    Minimal stand-in for the OpenAI client.

    responses.create returns `output_text` (a string, or a dict that is
    sent as JSON), or the `stream_events` when called with stream=True.
    Every call's arguments are recorded in `calls`.
    """

    def __init__(self, output_text="", stream_events=None, error=None):
        self.output_text = output_text
        self.stream_events = stream_events or []
        self.error = error
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error:
            raise self.error

        if kwargs.get("stream"):
            return iter(self.stream_events)

        text = self.output_text
        if isinstance(text, dict):
            text = json.dumps(text)

        return SimpleNamespace(output_text=text)


@pytest.fixture
def fake_client(monkeypatch):
    """Install a FakeClient; tweak its fields inside the test."""

    client = FakeClient()
    monkeypatch.setattr(config, "get_client", lambda: client)
    return client


def make_document(name, chunks, pages=None, embeddings=None, pdf_hash=None):
    """Build an in-memory Document without reading a PDF or calling an API."""

    pages = pages or [1] * len(chunks)

    if embeddings is None:
        # One-hot vectors: chunk i only matches itself. All documents
        # share one width, like real embeddings from the same model.
        embeddings = np.eye(len(chunks), EMBEDDING_DIM, dtype=np.float32)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    vectors = embeddings.copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return Document(
        path=None,
        name=name,
        pdf_hash=pdf_hash or f"hash-{name}",
        page_count=max(pages),
        chunks=list(chunks),
        chunk_pages=list(pages),
        embeddings=embeddings,
        index=index,
        bm25=BM25Okapi([tokenize(chunk) for chunk in chunks]),
    )


def make_pdf(path, pages_text):
    """Write a small, valid PDF with one line of text per page ('' = blank page)."""

    page_ids = [4 + 2 * i for i in range(len(pages_text))]

    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
            f"/Count {len(pages_text)} >>"
        ),
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for page_id, text in zip(page_ids, pages_text):
        stream = f"BT /F1 10 Tf 20 700 Td ({text}) Tj ET" if text else ""
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {page_id + 1} 0 R >>"
        )
        objects[page_id + 1] = (
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        )

    data = b"%PDF-1.4\n"
    offsets = {}

    for number in sorted(objects):
        offsets[number] = len(data)
        data += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1")

    xref_offset = len(data)
    size = max(objects) + 1

    data += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        data += f"{offsets[number]:010d} 00000 n \n".encode()
    data += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    path.write_bytes(data)
    return path
