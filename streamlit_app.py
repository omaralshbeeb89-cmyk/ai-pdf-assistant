"""
Web version of the AI PDF Assistant.

    streamlit run streamlit_app.py
"""

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from pdf_assistant import config, ingest, llm, retrieval
from pdf_assistant.library import build_collection, format_sources


st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="centered",
)


# ==================================================
# Cached loading
# ==================================================

@st.cache_data(show_spinner=False)
def pdf_fingerprint(path, mtime, size):
    """Hash a PDF once per (path, modified time, size)."""

    return ingest.file_hash(Path(path))


@st.cache_resource(show_spinner=False, max_entries=10)
def load_document(path, pdf_hash, _progress=None):
    """Load a PDF once and share it between browser sessions.
    pdf_hash is part of the cache key so an edited file is reloaded."""

    return ingest.load_document(Path(path), progress=_progress)


@st.cache_resource(show_spinner=False, max_entries=5)
def get_collection(pdf_hashes, _documents):
    """Combined index for a set of PDFs, keyed by their hashes."""

    return build_collection(_documents)


def available_pdfs():
    files = []

    for folder in (config.PDF_DIR, config.UPLOAD_DIR):
        if folder.is_dir():
            files.extend(sorted(folder.glob("*.pdf")))

    return [str(path) for path in files]


def save_upload(uploaded):
    config.UPLOAD_DIR.mkdir(exist_ok=True)

    target = config.UPLOAD_DIR / Path(uploaded.name).name
    target.write_bytes(uploaded.getbuffer())

    return target


# ==================================================
# Rendering helpers
# ==================================================

def render_sources(message, show_debug):
    sources = message.get("sources") or []

    if not sources:
        return

    multi_doc = message.get("multi_doc", False)
    results = [retrieval.SearchResult(**source) for source in sources]
    title = "Sources: " + ("" if multi_doc else "page ") + format_sources(results, multi_doc)

    with st.expander(title):

        if show_debug and message.get("query"):
            st.caption(f"Search query: {message['query']}")

        for source in sources:
            heading = f"{source['source']} · page {source['page']}" if multi_doc else f"Page {source['page']}"
            st.markdown(f"**{heading}**")

            if show_debug:
                faiss_part = (
                    f"FAISS #{source['faiss_rank']} ({source['faiss_score']:.3f})"
                    if source["faiss_rank"] is not None
                    else "FAISS: —"
                )
                rerank_part = (
                    f" · Rerank {source['rerank_score']:.0f}/10"
                    if source.get("rerank_score") is not None
                    else ""
                )
                st.caption(
                    f"RRF {source['rrf_score']:.4f} · {faiss_part} · "
                    f"BM25 {source['bm25_score']:.2f}{rerank_part}"
                )

            text = source["text"]
            st.text(text[:700] + ("…" if len(text) > 700 else ""))


def render_message(message, show_debug):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message, show_debug)


# ==================================================
# Sidebar
# ==================================================

st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("📄 Documents")

    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    pdfs = available_pdfs()

    # Save each new upload once and add it to the selection.
    if uploaded is not None and st.session_state.get("upload_id") != uploaded.file_id:
        saved = str(save_upload(uploaded))
        st.session_state.upload_id = uploaded.file_id
        pdfs = available_pdfs()
        selection = [p for p in st.session_state.get("pdf_choice", []) if p in pdfs]
        st.session_state.pdf_choice = selection + ([saved] if saved not in selection else [])

    if not pdfs:
        st.info("Upload a PDF to get started.")
        st.stop()

    # Drop files that no longer exist; pick a default the first time.
    selection = [p for p in st.session_state.get("pdf_choice", []) if p in pdfs]

    if "pdf_choice" not in st.session_state:
        default = str(config.DEFAULT_PDF)
        selection = [default if default in pdfs else pdfs[0]]

    st.session_state.pdf_choice = selection

    chosen = st.multiselect(
        "Search in",
        pdfs,
        key="pdf_choice",
        format_func=lambda path: Path(path).name,
        placeholder="Choose one or more PDFs",
    )

    use_rerank = st.toggle(
        "Rerank results",
        value=config.RERANK,
        help="The model re-scores the top 20 passages. Better ordering, about 1s slower.",
    )

    show_debug = st.toggle("Show retrieval details")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []


if not chosen:
    st.title("AI PDF Assistant")
    st.info("Choose at least one PDF in the sidebar.")
    st.stop()


# ==================================================
# Load the selected PDFs
# ==================================================

try:
    config.get_client()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

progress_area = st.empty()
documents = []

for number, choice in enumerate(chosen, start=1):
    path = Path(choice)
    stat = path.stat()
    pdf_hash = pdf_fingerprint(choice, stat.st_mtime, stat.st_size)

    def show_progress(done, total, name=path.name):
        progress_area.progress(
            done / total,
            text=f"Creating embeddings for {name}… {done} / {total} chunks",
        )

    with st.spinner(f"Loading {path.name} ({number}/{len(chosen)})… first time may take a minute"):
        try:
            documents.append(load_document(choice, pdf_hash, _progress=show_progress))
        except Exception as error:
            progress_area.empty()
            st.error(f"Could not load {path.name}: {error}")
            st.stop()

progress_area.empty()

pdf_hashes = tuple(sorted(doc.pdf_hash for doc in documents))

with st.spinner("Combining documents…"):
    collection = get_collection(pdf_hashes, documents)

# Start a fresh conversation when the selection changes.
if st.session_state.get("active_hashes") != pdf_hashes:
    st.session_state.active_hashes = pdf_hashes
    st.session_state.messages = []


# ==================================================
# Chat
# ==================================================

st.title("AI PDF Assistant")
st.caption(
    f"{', '.join(collection.names)} · {collection.page_count} pages · "
    f"{len(collection.chunks)} chunks"
)

for message in st.session_state.messages:
    render_message(message, show_debug)

placeholder = "Ask a question about the PDFs" if collection.is_multi else "Ask a question about the PDF"
question = st.chat_input(placeholder)

if question:
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message, show_debug)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching…"):
                query = llm.rewrite_question(question, history)
                results = retrieval.search(collection, query, rerank=use_rerank)

            if not results:
                answer = "No relevant passages were found."
                st.markdown(answer)
            else:
                answer = st.write_stream(
                    llm.stream_answer(question, results, history, collection.is_multi)
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "query": query,
                "multi_doc": collection.is_multi,
                "sources": [asdict(result) for result in results],
            }

            render_sources(assistant_message, show_debug)
            st.session_state.messages.append(assistant_message)

        except Exception as error:
            st.error(f"An error occurred: {error}")
            # Drop the unanswered question so history stays consistent.
            st.session_state.messages.pop()
