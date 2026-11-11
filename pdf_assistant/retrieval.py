"""Hybrid search: FAISS (meaning) + BM25 (keywords), merged with RRF."""

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from . import config
from .ingest import tokenize
from .library import Collection
from .rerank import rerank as rerank_results


@dataclass
class SearchResult:
    chunk_index: int
    source: str
    page: int
    text: str
    rrf_score: float
    faiss_rank: Optional[int]
    faiss_score: Optional[float]
    bm25_score: float
    rerank_score: Optional[float] = None


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
    collection: Collection,
    query,
    top_k=config.TOP_K,
    candidate_k=config.CANDIDATE_K,
    rerank=config.RERANK,
):
    """Return the top_k most relevant chunks for the query."""

    # With reranking, collect a wider pool for the model to choose from.
    pool_size = max(top_k, config.RERANK_CANDIDATES) if rerank else top_k

    if rerank:
        candidate_k = max(candidate_k, config.RERANK_CANDIDATES)

    candidate_k = min(candidate_k, len(collection.chunks))

    # Semantic search
    scores, indices = collection.index.search(_embed_query(query), candidate_k)

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
        keyword_scores = collection.bm25.get_scores(tokens)
    else:
        keyword_scores = np.zeros(len(collection.chunks))

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
    )[:pool_size]

    results = []

    for chunk_index, rrf_score in ranked:
        faiss_rank, faiss_score = semantic.get(chunk_index, (None, None))

        results.append(
            SearchResult(
                chunk_index=chunk_index,
                source=collection.chunk_sources[chunk_index],
                page=collection.chunk_pages[chunk_index],
                text=collection.chunks[chunk_index],
                rrf_score=rrf_score,
                faiss_rank=faiss_rank,
                faiss_score=faiss_score,
                bm25_score=float(keyword_scores[chunk_index]),
            )
        )

    if rerank:
        return rerank_results(query, results, top_k)

    return results
