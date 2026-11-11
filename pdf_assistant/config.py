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
DEFAULT_PDF = PDF_DIR / "ISLP_website.pdf"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

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
