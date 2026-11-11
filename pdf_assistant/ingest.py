"""Read a PDF, split it into chunks, embed them and build the search indexes."""

import hashlib
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from . import config


logger = logging.getLogger(__name__)

# pypdf logs harmless warnings about damaged PDF structure
# (e.g. "Ignoring wrong pointing object"). Hide them.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# faiss logs which CPU build it loaded; not useful to users.
logging.getLogger("faiss").setLevel(logging.WARNING)


@dataclass
class Document:
    """A processed PDF, ready to be searched."""

    path: Path
    name: str
    pdf_hash: str
    page_count: int
    chunks: list
    chunk_pages: list
    embeddings: np.ndarray
    index: faiss.Index
    bm25: BM25Okapi


# ==================================================
# Text helpers
# ==================================================

def tokenize(text):
    """Convert text into lowercase word tokens."""

    return re.findall(r"\b\w+\b", text.lower())


def split_into_chunks(text, chunk_size=None, overlap=None):
    """
    Split text into overlapping chunks of up to
    chunk_size characters without cutting words.
    """

    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap

    chunks = []
    length = len(text)
    start = 0

    while start < length:

        end = min(start + chunk_size, length)

        # Move the end back to the last whitespace,
        # unless that would make the chunk too short.
        if end < length:

            cut = max(
                text.rfind(" ", start, end),
                text.rfind("\n", start, end)
            )

            if cut > start + chunk_size // 2:
                end = cut

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        # Step back for overlap, then move forward
        # to the start of the next whole word.
        overlap_start = max(end - overlap, start + 1)
        next_start = overlap_start

        while (
            next_start < end
            and not text[next_start - 1].isspace()
        ):
            next_start += 1

        # No whitespace in the overlap: keep the hard cut
        # so the overlap is not lost.
        if next_start >= end:
            next_start = overlap_start

        start = next_start

    return chunks


def file_hash(path):
    """SHA-256 of a file, read in blocks."""

    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


# ==================================================
# Cache files
# ==================================================

def _cache_paths(name):
    config.CACHE_DIR.mkdir(exist_ok=True)

    return (
        config.CACHE_DIR / f"{name}_embeddings.npz",
        config.CACHE_DIR / f"{name}_faiss_index.bin",
        config.CACHE_DIR / f"{name}_faiss_index_hash.txt",
    )


def _load_chunk_cache(cache_path, pdf_hash):
    """Return (chunks, pages, embeddings) from the cache, or None."""

    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cache:

            up_to_date = (
                str(cache["pdf_hash"].item()) == pdf_hash
                and str(cache["embedding_model"].item()) == config.EMBEDDING_MODEL
                and int(cache["chunk_size"].item()) == config.CHUNK_SIZE
                and int(cache["chunk_overlap"].item()) == config.CHUNK_OVERLAP
                and str(cache["chunking_version"].item()) == config.CHUNKING_VERSION
            )

            if not up_to_date:
                logger.info("Cache is out of date. Rebuilding...")
                return None

            chunks = cache["chunks"].tolist()
            pages = cache["chunk_pages"]
            embeddings = np.asarray(cache["embeddings"], dtype=np.float32)

    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        logger.warning("Cache is unreadable. Rebuilding...")
        return None

    valid = (
        len(chunks) > 0
        and len(chunks) == len(pages) == len(embeddings)
        and embeddings.ndim == 2
        and embeddings.shape[1] > 0
        and np.isfinite(embeddings).all()
        and np.all(pages >= 1)
    )

    if not valid:
        logger.warning("Cache is invalid. Rebuilding...")
        return None

    return chunks, [int(page) for page in pages], embeddings


def _save_chunk_cache(cache_path, pdf_hash, chunks, pages, embeddings):
    np.savez_compressed(
        cache_path,
        pdf_hash=np.asarray(pdf_hash),
        embedding_model=np.asarray(config.EMBEDDING_MODEL),
        chunk_size=np.asarray(config.CHUNK_SIZE),
        chunk_overlap=np.asarray(config.CHUNK_OVERLAP),
        chunking_version=np.asarray(config.CHUNKING_VERSION),
        chunks=np.asarray(chunks, dtype=str),
        chunk_pages=np.asarray(pages, dtype=np.int32),
        embeddings=embeddings,
    )


# ==================================================
# Building steps
# ==================================================

def _extract_chunks(reader):
    """Split every page into chunks, remembering each chunk's page."""

    chunks = []
    pages = []
    empty_pages = 0

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = (page.extract_text() or "").strip()

        if not page_text:
            empty_pages += 1
            continue

        for chunk in split_into_chunks(page_text):
            chunks.append(chunk)
            pages.append(page_number)

    if empty_pages:
        logger.warning(
            "%d of %d pages had no extractable text "
            "(they may be scanned images).",
            empty_pages,
            len(reader.pages),
        )

    if not chunks:
        raise ValueError("No extractable text found in the PDF.")

    return chunks, pages


def _embed(chunks, progress=None):
    """Create embeddings in batches. progress(done, total) is optional."""

    client = config.get_client()
    vectors = []

    for start in range(0, len(chunks), config.BATCH_SIZE):

        batch = chunks[start:start + config.BATCH_SIZE]

        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )

        vectors.extend(
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        )

        if progress:
            progress(len(vectors), len(chunks))

    embeddings = np.asarray(vectors, dtype=np.float32)

    if len(embeddings) != len(chunks) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid or mismatched embeddings.")

    return embeddings


def _index_fingerprint(pdf_hash, embeddings):
    hasher = hashlib.sha256()

    for part in (
        pdf_hash,
        config.EMBEDDING_MODEL,
        str(config.CHUNK_SIZE),
        str(config.CHUNK_OVERLAP),
        config.CHUNKING_VERSION,
    ):
        hasher.update(part.encode("utf-8"))

    hasher.update(embeddings.tobytes())

    return hasher.hexdigest()


def _load_or_build_index(embeddings, fingerprint, index_path, hash_path):
    """Load the saved FAISS index if it matches, otherwise rebuild it."""

    if index_path.exists() and hash_path.exists():

        try:
            if hash_path.read_text(encoding="utf-8").strip() == fingerprint:

                index = faiss.read_index(str(index_path))

                if (
                    index.ntotal == len(embeddings)
                    and index.d == embeddings.shape[1]
                ):
                    logger.info("Loaded saved FAISS index.")
                    return index

        except Exception as error:
            logger.warning("Could not load FAISS index: %s", error)

    logger.info("Building FAISS index...")

    vectors = embeddings.copy()
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(index_path))
    hash_path.write_text(fingerprint, encoding="utf-8")

    return index


# ==================================================
# Public entry point
# ==================================================

def load_document(pdf_path, progress=None):
    """
    Load a PDF and everything needed to search it.
    Uses the cache when possible; otherwise creates embeddings
    (calling progress(done, total) after each batch).
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_hash = file_hash(pdf_path)
    name = pdf_path.stem
    cache_path, index_path, hash_path = _cache_paths(name)

    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)

    cached = _load_chunk_cache(cache_path, pdf_hash)

    if cached:
        chunks, pages, embeddings = cached
        logger.info("Loaded %d chunks from cache.", len(chunks))

    else:
        logger.info("Reading %s...", pdf_path.name)
        chunks, pages = _extract_chunks(reader)

        logger.info("Creating embeddings for %d chunks...", len(chunks))
        embeddings = _embed(chunks, progress)

        _save_chunk_cache(cache_path, pdf_hash, chunks, pages, embeddings)
        logger.info("Embeddings cache saved.")

    index = _load_or_build_index(
        embeddings,
        _index_fingerprint(pdf_hash, embeddings),
        index_path,
        hash_path,
    )

    bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])

    return Document(
        path=pdf_path,
        name=pdf_path.name,
        pdf_hash=pdf_hash,
        page_count=page_count,
        chunks=chunks,
        chunk_pages=pages,
        embeddings=embeddings,
        index=index,
        bm25=bm25,
    )
