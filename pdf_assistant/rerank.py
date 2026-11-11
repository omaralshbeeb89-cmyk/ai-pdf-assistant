"""Rerank search candidates by asking the chat model to score each one."""

import json
import logging

from . import config


logger = logging.getLogger(__name__)


SCORES_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                },
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def rerank(query, results, top_k):
    """
    Score every candidate 0-10 for relevance to the query and
    return the best top_k. Ties keep the original (RRF) order.
    If the model call fails, the original order is used.
    """

    if not results:
        return results

    passages = "\n\n".join(
        f'<passage id="{i}">\n{result.text}\n</passage>'
        for i, result in enumerate(results)
    )

    prompt = f"""
You are ranking passages for a search engine over a PDF.

Rate how useful each passage is for answering the search query:
10 = directly answers it, 5 = related but incomplete, 0 = irrelevant.
Rate every passage exactly once, using its id.

SEARCH QUERY:
{query}

PASSAGES:
{passages}
"""

    try:
        response = config.get_client().responses.create(
            model=config.RERANK_MODEL,
            input=prompt,
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "passage_scores",
                    "schema": SCORES_SCHEMA,
                    "strict": True,
                }
            },
        )

        data = json.loads(response.output_text)

        scores = {
            int(item["id"]): float(item["score"])
            for item in data["scores"]
            if 0 <= int(item["id"]) < len(results)
        }

    except Exception as error:
        logger.warning("Reranking failed, using hybrid order: %s", error)
        return results[:top_k]

    for i, result in enumerate(results):
        result.rerank_score = scores.get(i, -1.0)

    # sorted() is stable, so equal scores keep the RRF order.
    ranked = sorted(results, key=lambda result: -result.rerank_score)

    return ranked[:top_k]
