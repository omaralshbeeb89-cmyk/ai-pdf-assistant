"""Hybrid search: FAISS (meaning) + BM25 (keywords), merged with RRF."""

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from . import config
from .ingest import Document, tokenize


@dataclass
class SearchResult:
    chunk_index: int
    page: int
    text: str
    rrf_score: float
    faiss_rank: Optional[int]
    faiss_score: Optional[float]
    bm25_score: float


def _embed_query(query):
    response = config.get_client().embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=query,
    )

    vector = np.asarray(
        response.data[0].embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    faiss.normalize_L2(vector)

    return vector


def search(
    doc: Document,
    query,
    top_k=config.TOP_K,
    candidate_k=config.CANDIDATE_K,
):
    """Return the top_k most relevant chunks for the query."""

    candidate_k = min(candidate_k, len(doc.chunks))

    # Semantic search
    scores, indices = doc.index.search(_embed_query(query), candidate_k)

    semantic = {
        int(chunk_index): (rank, float(score))
        for rank, (chunk_index, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        )
        if chunk_index >= 0
    }

    # Keyword search: only chunks that actually match.
    tokens = tokenize(query)

    if tokens:
        keyword_scores = doc.bm25.get_scores(tokens)
    else:
        keyword_scores = np.zeros(len(doc.chunks))

    positive = np.flatnonzero(keyword_scores > 0)

    keyword_indices = positive[
        np.argsort(keyword_scores[positive])[::-1][:candidate_k]
    ]

    # Reciprocal Rank Fusion
    rrf_scores = {}

    for chunk_index, (rank, _) in semantic.items():
        rrf_scores[chunk_index] = (
            rrf_scores.get(chunk_index, 0.0) + 1.0 / (config.RRF_K + rank)
        )

    for rank, chunk_index in enumerate(keyword_indices, start=1):
        chunk_index = int(chunk_index)
        rrf_scores[chunk_index] = (
            rrf_scores.get(chunk_index, 0.0) + 1.0 / (config.RRF_K + rank)
        )

    ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    results = []

    for chunk_index, rrf_score in ranked:
        faiss_rank, faiss_score = semantic.get(chunk_index, (None, None))

        results.append(
            SearchResult(
                chunk_index=chunk_index,
                page=doc.chunk_pages[chunk_index],
                text=doc.chunks[chunk_index],
                rrf_score=rrf_score,
                faiss_rank=faiss_rank,
                faiss_score=faiss_score,
                bm25_score=float(keyword_scores[chunk_index]),
            )
        )

    return results
