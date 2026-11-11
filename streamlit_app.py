"""
Web version of the AI PDF Assistant.

    streamlit run streamlit_app.py
"""

import threading
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from pdf_assistant import config, conversations, ingest, llm, retrieval, usage
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


MAX_LOADED_DOCUMENTS = 10


@st.cache_resource(show_spinner=False)
def document_store():
    """Loaded PDFs shared by all browser sessions, plus a lock.

    st.cache_resource is not used on the loader itself: it would record
    the progress-bar updates made while embedding a new PDF and fail,
    because the progress bar is created outside the cached function."""

    return {"documents": {}, "lock": threading.Lock()}


def load_document(path, pdf_hash, progress=None):
    """Load a PDF once; pdf_hash in the key reloads an edited file."""

    store = document_store()
    key = (path, pdf_hash)

    # The lock stops two sessions from embedding the same PDF at once.
    with store["lock"]:
        documents = store["documents"]

        if key not in documents:
            documents[key] = ingest.load_document(Path(path), progress=progress)

            # Forget the oldest PDF when too many are loaded.
            while len(documents) > MAX_LOADED_DOCUMENTS:
                documents.pop(next(iter(documents)))

        return documents[key]


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
# Conversation actions (run as button callbacks,
# before the page is drawn)
# ==================================================

def start_new_conversation():
    st.session_state.conversation = conversations.new_conversation()


def open_conversation(conversation_id):
    try:
        conversation = conversations.load(conversation_id)
    except Exception as error:
        st.session_state.notice = f"Could not open the conversation: {error}"
        return

    st.session_state.conversation = conversation
    st.session_state.pdf_choice = [
        path for path in conversation["pdf_paths"] if Path(path).is_file()
    ]


def delete_current_conversation():
    conversations.delete(st.session_state.conversation["id"])
    start_new_conversation()


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


def render_usage(message, show_debug):
    data = message.get("usage")

    if not data:
        return

    st.caption(usage.format_usage(data))

    if show_debug:
        for call in data["calls"]:
            st.caption(
                f"· {call['purpose']}: {call['input_tokens']:,} in / "
                f"{call['output_tokens']:,} out · {usage.format_cost(call['cost'])}"
            )


def render_message(message, show_debug):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message, show_debug)
            render_usage(message, show_debug)


# ==================================================
# Sidebar: documents and options
# ==================================================

st.session_state.setdefault("conversation", conversations.new_conversation())

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

    # Drop files that no longer exist; pick a default the first time.
    selection = [p for p in st.session_state.get("pdf_choice", []) if p in pdfs]

    if "pdf_choice" not in st.session_state and pdfs:
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

    show_debug = st.toggle("Show retrieval details and cost breakdown")

    st.divider()
    st.header("💬 Conversations")

    # Filled at the end of the script, so it includes the latest answer.
    conversations_area = st.container()


# ==================================================
# Main area
# ==================================================

def chat_page():
    conversation = st.session_state.conversation

    st.title("AI PDF Assistant")

    if notice := st.session_state.pop("notice", None):
        st.warning(notice)

    # A saved conversation whose PDFs are gone can be read, not continued.
    missing = [
        Path(path).name
        for path in conversation["pdf_paths"]
        if not Path(path).is_file()
    ]

    if conversation["messages"] and missing:
        for message in conversation["messages"]:
            render_message(message, show_debug)

        st.warning(
            f"This conversation used {', '.join(missing)}, which is no longer "
            "available. You can read it, but start a new conversation to ask more."
        )
        return

    if not pdfs:
        st.info("Upload a PDF in the sidebar to get started.")
        return

    if not chosen:
        st.info("Choose at least one PDF in the sidebar.")
        return

    try:
        config.get_client()
    except RuntimeError as error:
        st.error(str(error))
        return

    # ---------- load the selected PDFs ----------

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
                documents.append(load_document(choice, pdf_hash, progress=show_progress))
            except Exception as error:
                progress_area.empty()
                st.error(f"Could not load {path.name}: {error}")
                return

    progress_area.empty()

    pdf_hashes = tuple(sorted(doc.pdf_hash for doc in documents))

    with st.spinner("Combining documents…"):
        collection = get_collection(pdf_hashes, documents)

    # A conversation belongs to its PDFs: changing the selection
    # starts a new one (the old one stays saved).
    if conversation["messages"] and tuple(sorted(conversation["pdf_hashes"])) != pdf_hashes:
        start_new_conversation()
        conversation = st.session_state.conversation

    conversation["pdf_paths"] = list(chosen)
    conversation["pdf_hashes"] = list(pdf_hashes)

    # ---------- chat ----------

    st.caption(
        f"{', '.join(collection.names)} · {collection.page_count} pages · "
        f"{len(collection.chunks)} chunks"
    )

    for message in conversation["messages"]:
        render_message(message, show_debug)

    placeholder = "Ask a question about the PDFs" if collection.is_multi else "Ask a question about the PDF"
    question = st.chat_input(placeholder)

    if not question:
        return

    history = conversations.history(conversation)

    user_message = {"role": "user", "content": question}
    render_message(user_message, show_debug)

    with st.chat_message("assistant"):
        try:
            with usage.track() as tracked:
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
                "usage": tracked.to_dict(),
            }

            render_sources(assistant_message, show_debug)
            render_usage(assistant_message, show_debug)

        except Exception as error:
            st.error(f"An error occurred: {error}")
            return

    conversation["messages"].extend([user_message, assistant_message])

    try:
        conversations.save(conversation)
    except OSError as error:
        st.warning(f"The answer could not be saved: {error}")


chat_page()


# ==================================================
# Sidebar: conversation list and costs
# ==================================================

with conversations_area:
    current = st.session_state.conversation

    st.button(
        "➕ New conversation",
        on_click=start_new_conversation,
        use_container_width=True,
    )

    saved = conversations.list_conversations()

    for item in saved[:30]:
        st.button(
            item["title"],
            key=f"conversation-{item['id']}",
            on_click=open_conversation,
            args=(item["id"],),
            type="primary" if item["id"] == current["id"] else "secondary",
            help=f"{item['questions']} question(s) · last used {item['updated'].replace('T', ' ')}",
            use_container_width=True,
        )

    if current["messages"]:
        with st.popover("🗑 Delete this conversation", use_container_width=True):
            st.write("This cannot be undone.")
            st.button("Delete", type="primary", on_click=delete_current_conversation)

    st.divider()

    if current["messages"]:
        st.caption(f"This conversation: {usage.format_usage(conversations.conversation_usage(current))}")

    if saved:
        all_cost = sum(item["cost"] for item in saved)
        st.caption(f"All saved conversations: {usage.format_cost(all_cost)}")
