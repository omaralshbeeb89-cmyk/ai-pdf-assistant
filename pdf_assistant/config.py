"""Settings shared by the whole project."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Paths are absolute so the app works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_DIR / "pdf"
UPLOAD_DIR = PROJECT_DIR / "uploads"
CACHE_DIR = PROJECT_DIR / "cache"
CONVERSATIONS_DIR = PROJECT_DIR / "conversations"
DEFAULT_PDF = PDF_DIR / "ISLP_website.pdf"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# Estimated prices in US$ per 1 million tokens: (input, output).
# Used only to show approximate costs. Check
# https://openai.com/api/pricing and update if they change.
PRICES_PER_MILLION = {
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
}

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100

# Bump this when the chunking logic changes so
# old caches are rebuilt automatically.
CHUNKING_VERSION = "word-boundary-v1"

# Retrieval
TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60

# Reranking: the hybrid search collects RERANK_CANDIDATES passages,
# then the model scores them and the best TOP_K are kept.
# Off by default: on the ISLP test set (40 questions) it raised MRR
# from 0.80 to 0.86 but lowered page hit@5 from 97.5% to 92.5% and
# doubled search time. Re-run the evaluation before changing this.
RERANK = False
RERANK_MODEL = CHAT_MODEL
RERANK_CANDIDATES = 20

# Keep the last 3 conversation turns.
MAX_HISTORY_MESSAGES = 6


if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be positive.")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE.")


@lru_cache(maxsize=1)
def get_client():
    """Create the OpenAI client once, on first use."""

    load_dotenv(PROJECT_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to the .env file "
            f"in {PROJECT_DIR}."
        )

    return OpenAI(api_key=api_key)
