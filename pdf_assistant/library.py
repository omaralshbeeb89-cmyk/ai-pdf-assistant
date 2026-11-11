"""Combine several processed PDFs into one searchable collection."""

from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import tokenize


@dataclass
class Collection:
    """One or more PDFs searched together through a single index."""

    documents: list
    chunks: list
    chunk_pages: list
    chunk_sources: list  # file name of each chunk
    index: faiss.Index
    bm25: BM25Okapi

    @property
    def is_multi(self):
        return len(self.documents) > 1

    @property
    def page_count(self):
        return sum(doc.page_count for doc in self.documents)

    @property
    def names(self):
        return [doc.name for doc in self.documents]


def build_collection(documents):
    """
    Build one collection from loaded Documents.

    A single PDF reuses its own indexes. Several PDFs get a combined
    FAISS index and BM25 index, so scores are comparable across files
    and the most relevant passages win regardless of which PDF they
    come from.
    """

    # The same file can be selected twice (e.g. pdf/ and uploads/).
    # Keep the first copy so the order of the documents is preserved.
    unique = []
    seen = set()

    for doc in documents:
        if doc.pdf_hash not in seen:
            seen.add(doc.pdf_hash)
            unique.append(doc)

    if not unique:
        raise ValueError("Select at least one PDF.")

    if len(unique) == 1:
        doc = unique[0]
        return Collection(
            documents=unique,
            chunks=doc.chunks,
            chunk_pages=doc.chunk_pages,
            chunk_sources=[doc.name] * len(doc.chunks),
            index=doc.index,
            bm25=doc.bm25,
        )

    chunks, pages, sources = [], [], []

    for doc in unique:
        chunks.extend(doc.chunks)
        pages.extend(doc.chunk_pages)
        sources.extend([doc.name] * len(doc.chunks))

    vectors = np.vstack([doc.embeddings for doc in unique]).astype(np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])

    return Collection(
        documents=unique,
        chunks=chunks,
        chunk_pages=pages,
        chunk_sources=sources,
        index=index,
        bm25=bm25,
    )


def format_sources(results, multi_doc):
    """'252, 254'  or  'ISLP_website.pdf p. 252, 254; notes.pdf p. 12'"""

    by_source = {}

    for result in results:
        by_source.setdefault(result.source, set()).add(result.page)

    if not multi_doc:
        pages = sorted(set().union(*by_source.values()))
        return ", ".join(map(str, pages))

    return "; ".join(
        f"{source} p. {', '.join(map(str, sorted(pages)))}"
        for source, pages in by_source.items()
    )
