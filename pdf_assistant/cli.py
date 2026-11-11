"""Terminal version of the assistant."""

import argparse
import logging
import sys

from . import config, ingest, llm, retrieval


def _print_debug(query, results):
    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL DEBUG")
    print("Search query:", query)
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult #{rank}")
        print("Chunk index:", result.chunk_index)
        print("PDF page:", result.page)
        print("RRF score:", f"{result.rrf_score:.6f}")

        if result.faiss_rank is not None:
            print("FAISS rank:", result.faiss_rank)
            print("FAISS similarity:", f"{result.faiss_score:.4f}")
        else:
            print("FAISS: Not in semantic candidates")

        print("BM25 score:", f"{result.bm25_score:.4f}")
        print("\nRetrieved text:")
        print(result.text[:500])
        print("-" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")

    parser.add_argument(
        "pdf",
        nargs="?",
        default=str(config.DEFAULT_PDF),
        help="Path to the PDF file (default: pdf/ISLP_website.pdf)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieval details before each answer",
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
        doc = ingest.load_document(
            args.pdf,
            progress=lambda done, total: print(f"Processed {done} / {total} chunks"),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1

    print(f"\nAI PDF ASSISTANT READY  ({doc.name}, {len(doc.chunks)} chunks)")
    print("Commands:")
    print("  exit  - Close the assistant")
    print("  reset - Clear conversation memory")

    history = []

    while True:

        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if question.lower() == "reset":
            history.clear()
            print("Conversation memory cleared.")
            continue

        try:
            query = llm.rewrite_question(question, history)
            results = retrieval.search(doc, query)

            if not results:
                print("No relevant chunks found.")
                continue

            if args.debug:
                _print_debug(query, results)

            print("\nAI PDF ASSISTANT:\n")

            parts = []

            for piece in llm.stream_answer(question, results, history):
                print(piece, end="", flush=True)
                parts.append(piece)

            answer = "".join(parts).strip()

            pages = sorted({result.page for result in results})
            print("\n\nRetrieved source pages:", ", ".join(map(str, pages)))

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            history = llm.trim_history(history)

        except KeyboardInterrupt:
            print("\n(Answer interrupted.)")

        except Exception as error:
            print(f"\nAn error occurred: {error}")

    return 0
