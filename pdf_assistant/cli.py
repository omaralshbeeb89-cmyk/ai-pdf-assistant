"""Terminal version of the assistant."""

import argparse
import logging
import sys
from dataclasses import asdict

from . import config, conversations, ingest, llm, retrieval, usage
from .library import build_collection, format_sources


def _print_debug(query, results):
    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL DEBUG")
    print("Search query:", query)
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult #{rank}")
        print("Chunk index:", result.chunk_index)
        print("Source:", result.source)
        print("PDF page:", result.page)
        print("RRF score:", f"{result.rrf_score:.6f}")

        if result.faiss_rank is not None:
            print("FAISS rank:", result.faiss_rank)
            print("FAISS similarity:", f"{result.faiss_score:.4f}")
        else:
            print("FAISS: Not in semantic candidates")

        print("BM25 score:", f"{result.bm25_score:.4f}")

        if result.rerank_score is not None:
            print("Rerank score:", f"{result.rerank_score:.0f} / 10")
        print("\nRetrieved text:")
        print(result.text[:500])
        print("-" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask questions about one or more PDFs.")

    parser.add_argument(
        "pdfs",
        nargs="*",
        default=[str(config.DEFAULT_PDF)],
        metavar="pdf",
        help="PDF files to search together (default: pdf/ISLP_website.pdf)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieval details before each answer",
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        default=config.RERANK,
        help="Let the model re-score search results (slower)",
    )

    args = parser.parse_args(argv)

    # Windows may use a legacy code page that cannot print
    # symbols like λ or ε, which appear in many PDFs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Hide the OpenAI client's per-request log lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        documents = []

        for pdf in args.pdfs:
            print(f"Loading {pdf}...")
            documents.append(
                ingest.load_document(
                    pdf,
                    progress=lambda done, total: print(f"Processed {done} / {total} chunks"),
                )
            )

        collection = build_collection(documents)

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1

    print(
        f"\nAI PDF ASSISTANT READY  "
        f"({', '.join(collection.names)}; {len(collection.chunks)} chunks)"
    )
    print("Commands:")
    print("  exit  - Close the assistant")
    print("  reset - Start a new conversation")
    print("Conversations are saved and also appear in the web app.")

    def start_conversation():
        return conversations.new_conversation(
            [str(doc.path.resolve()) for doc in collection.documents],
            [doc.pdf_hash for doc in collection.documents],
        )

    conversation = start_conversation()
    session_usage = []

    while True:

        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            question = "exit"
            print()

        if not question:
            continue

        if question.lower() == "exit":
            if session_usage:
                print(f"Session total: {usage.format_usage(usage.total(session_usage))}")
            print("Goodbye!")
            break

        if question.lower() == "reset":
            conversation = start_conversation()
            print("Started a new conversation.")
            continue

        history = conversations.history(conversation)

        try:
            with usage.track() as tracked:
                query = llm.rewrite_question(question, history)
                results = retrieval.search(collection, query, rerank=args.rerank)

                if not results:
                    print("No relevant chunks found.")
                    continue

                if args.debug:
                    _print_debug(query, results)

                print("\nAI PDF ASSISTANT:\n")

                parts = []

                for piece in llm.stream_answer(
                    question, results, history, collection.is_multi
                ):
                    print(piece, end="", flush=True)
                    parts.append(piece)

            answer = "".join(parts).strip()
            cost = tracked.to_dict()
            session_usage.append(cost)

            print(
                "\n\nRetrieved sources:",
                format_sources(results, collection.is_multi),
            )
            print(f"[{usage.format_usage(cost)}]")

            conversation["messages"].append({"role": "user", "content": question})
            conversation["messages"].append({
                "role": "assistant",
                "content": answer,
                "query": query,
                "multi_doc": collection.is_multi,
                "sources": [asdict(result) for result in results],
                "usage": cost,
            })
            conversations.save(conversation)

        except KeyboardInterrupt:
            print("\n(Answer interrupted.)")

        except Exception as error:
            print(f"\nAn error occurred: {error}")

    return 0
