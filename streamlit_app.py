"""
Web version of the AI PDF Assistant.

    streamlit run streamlit_app.py
"""

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from pdf_assistant import config, ingest, llm, retrieval


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


@st.cache_resource(show_spinner=False, max_entries=5)
def load_document(path, pdf_hash, _progress=None):
    """Load a PDF once and share it between browser sessions.
    pdf_hash is part of the cache key so an edited file is reloaded."""

    return ingest.load_document(Path(path), progress=_progress)


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

    pages = sorted({source["page"] for source in sources})

    with st.expander("Sources: page " + ", ".join(map(str, pages))):

        if show_debug and message.get("query"):
            st.caption(f"Search query: {message['query']}")

        for source in sources:
            st.markdown(f"**Page {source['page']}**")

            if show_debug:
                faiss_part = (
                    f"FAISS #{source['faiss_rank']} ({source['faiss_score']:.3f})"
                    if source["faiss_rank"] is not None
                    else "FAISS: —"
                )
                st.caption(
                    f"RRF {source['rrf_score']:.4f} · {faiss_part} · "
                    f"BM25 {source['bm25_score']:.2f}"
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
    st.header("📄 Document")

    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    # Save each new upload once and select it.
    if uploaded is not None and st.session_state.get("upload_id") != uploaded.file_id:
        saved = save_upload(uploaded)
        st.session_state.upload_id = uploaded.file_id
        st.session_state.pdf_choice = str(saved)

    pdfs = available_pdfs()

    if not pdfs:
        st.info("Upload a PDF to get started.")
        st.stop()

    if st.session_state.get("pdf_choice") not in pdfs:
        default = str(config.DEFAULT_PDF)
        st.session_state.pdf_choice = default if default in pdfs else pdfs[0]

    choice = st.selectbox(
        "Choose a PDF",
        pdfs,
        key="pdf_choice",
        format_func=lambda path: Path(path).name,
    )

    show_debug = st.toggle("Show retrieval details")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []


# ==================================================
# Load the selected PDF
# ==================================================

try:
    config.get_client()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

path = Path(choice)
stat = path.stat()
pdf_hash = pdf_fingerprint(choice, stat.st_mtime, stat.st_size)

progress_area = st.empty()


def show_progress(done, total):
    progress_area.progress(
        done / total,
        text=f"Creating embeddings… {done} / {total} chunks",
    )


with st.spinner(f"Loading {path.name}… (first time may take a minute)"):
    try:
        doc = load_document(choice, pdf_hash, _progress=show_progress)
    except Exception as error:
        progress_area.empty()
        st.error(f"Could not load the PDF: {error}")
        st.stop()

progress_area.empty()

# Start a fresh conversation when the document changes.
if st.session_state.get("active_hash") != doc.pdf_hash:
    st.session_state.active_hash = doc.pdf_hash
    st.session_state.messages = []


# ==================================================
# Chat
# ==================================================

st.title("AI PDF Assistant")
st.caption(f"{doc.name} · {doc.page_count} pages · {len(doc.chunks)} chunks")

for message in st.session_state.messages:
    render_message(message, show_debug)

question = st.chat_input("Ask a question about the PDF")

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
            with st.spinner("Searching the PDF…"):
                query = llm.rewrite_question(question, history)
                results = retrieval.search(doc, query)

            if not results:
                answer = "No relevant passages were found in the PDF."
                st.markdown(answer)
            else:
                answer = st.write_stream(
                    llm.stream_answer(question, results, history)
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "query": query,
                "sources": [asdict(result) for result in results],
            }

            render_sources(assistant_message, show_debug)
            st.session_state.messages.append(assistant_message)

        except Exception as error:
            st.error(f"An error occurred: {error}")
            # Drop the unanswered question so history stays consistent.
            st.session_state.messages.pop()
