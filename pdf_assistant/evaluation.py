"""
Measure how well the search finds the right page.

1. Generate test questions from random passages of a PDF
   (the passage's page is the expected answer):

       python -m pdf_assistant.evaluation generate [pdf] [--n 40]

2. Run the questions through the search and report scores:

       python -m pdf_assistant.evaluation run [pdf] [--compare]

   --compare runs both with and without reranking.

Metrics
  Page hit@5   the expected page is among the 5 retrieved passages
  Page MRR     1 / rank of the first passage from the expected page
  Chunk hit@5  the exact source passage was retrieved
"""

import argparse
import json
import logging
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, ingest, retrieval
from .library import build_collection


logger = logging.getLogger(__name__)

EVAL_DIR = config.PROJECT_DIR / "eval"
WORKERS = 8

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
    },
    "required": ["usable", "question"],
    "additionalProperties": False,
}


def questions_path(doc):
    return EVAL_DIR / f"{Path(doc.name).stem}_questions.json"


# ==================================================
# Generate questions
# ==================================================

def _question_for(passage):
    """Ask the model for one question this passage answers, or None."""

    prompt = f"""
You are creating a test set for a search engine over a textbook.

Write ONE question that a student might ask and that this passage answers.
- Use your own words; do not copy long phrases from the passage.
- The question must make sense on its own, without seeing the passage.
- Do not mention "the passage" or "the text".

If the passage is not useful content (table of contents, index,
bibliography, exercises list, mostly code or numbers), set usable to false.

PASSAGE:
{passage}
"""

    response = config.get_client().responses.create(
        model=config.CHAT_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "test_question",
                "schema": QUESTION_SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(response.output_text)

    if not data["usable"] or not data["question"].strip():
        return None

    return data["question"].strip()


def generate(doc, n, seed=42):
    rng = random.Random(seed)

    candidates = [
        index for index, chunk in enumerate(doc.chunks)
        if len(chunk) >= 600
    ]

    # Sample extra passages because some will be marked unusable.
    picked = rng.sample(candidates, min(n * 2, len(candidates)))

    with ThreadPoolExecutor(WORKERS) as pool:
        questions = list(
            pool.map(lambda index: _question_for(doc.chunks[index]), picked)
        )

    items = [
        {
            "question": question,
            "page": doc.chunk_pages[index],
            "chunk_index": index,
        }
        for index, question in zip(picked, questions)
        if question
    ][:n]

    EVAL_DIR.mkdir(exist_ok=True)
    path = questions_path(doc)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(items)} questions to {path}")


# ==================================================
# Run the evaluation
# ==================================================

def _evaluate_one(collection, source, item, rerank):
    start = time.perf_counter()
    results = retrieval.search(collection, item["question"], rerank=rerank)
    seconds = time.perf_counter() - start

    # A hit must come from the right PDF, not just the right page number.
    hits = [
        result.source == source and result.page == item["page"]
        for result in results
    ]

    return {
        **item,
        "retrieved": [f"{result.source} p.{result.page}" for result in results],
        "page_rank": hits.index(True) + 1 if True in hits else None,
        # The question PDF is first in the collection, so its chunk
        # numbers are unchanged even when other PDFs are added.
        "chunk_hit": item["chunk_index"] in [result.chunk_index for result in results],
        "seconds": seconds,
    }


def evaluate(collection, items, rerank):
    source = collection.documents[0].name

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(
            pool.map(
                lambda item: _evaluate_one(collection, source, item, rerank),
                items,
            )
        )

    total = len(rows)

    return {
        "rerank": rerank,
        "questions": total,
        "page_hit": sum(row["page_rank"] is not None for row in rows) / total,
        "page_mrr": sum(1 / row["page_rank"] for row in rows if row["page_rank"]) / total,
        "chunk_hit": sum(row["chunk_hit"] for row in rows) / total,
        "seconds": statistics.mean(row["seconds"] for row in rows),
        "misses": [row for row in rows if row["page_rank"] is None],
    }


def print_report(reports):
    print()
    print(f"{'Setting':<16}{'Page hit@5':>12}{'Page MRR':>11}{'Chunk hit@5':>13}{'Avg time':>11}")
    print("-" * 63)

    for report in reports:
        name = "hybrid + rerank" if report["rerank"] else "hybrid only"
        print(
            f"{name:<16}"
            f"{report['page_hit']:>12.1%}"
            f"{report['page_mrr']:>11.3f}"
            f"{report['chunk_hit']:>13.1%}"
            f"{report['seconds']:>10.1f}s"
        )

    print(f"\n{reports[0]['questions']} questions.")

    last = reports[-1]

    if last["misses"]:
        print(f"\nMissed questions ({'with' if last['rerank'] else 'without'} rerank):")
        for row in last["misses"]:
            print(f"  - page {row['page']}: {row['question']}")
            print(f"    got {row['retrieved']}")


# ==================================================
# Command line
# ==================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("command", choices=["generate", "run"])
    parser.add_argument("pdf", nargs="?", default=str(config.DEFAULT_PDF))
    parser.add_argument("--n", type=int, default=40, help="questions to generate")
    parser.add_argument("--compare", action="store_true", help="run with and without reranking")
    parser.add_argument("--rerank", action="store_true", default=config.RERANK, help="run with reranking")
    parser.add_argument(
        "--with",
        dest="extra",
        nargs="+",
        default=[],
        metavar="PDF",
        help="also load these PDFs, to test search when several files are selected",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    doc = ingest.load_document(args.pdf)

    if args.command == "generate":
        generate(doc, args.n)
        return 0

    path = questions_path(doc)

    if not path.exists():
        print(f"No questions yet. Run first:\n  python -m pdf_assistant.evaluation generate {args.pdf}")
        return 1

    items = json.loads(path.read_text(encoding="utf-8"))

    collection = build_collection(
        [doc] + [ingest.load_document(pdf) for pdf in args.extra]
    )

    if collection.is_multi:
        print("Searching across:", ", ".join(collection.names))

    settings = [False, True] if args.compare else [args.rerank]

    print_report([evaluate(collection, items, rerank) for rerank in settings])
    return 0


if __name__ == "__main__":
    sys.exit(main())
