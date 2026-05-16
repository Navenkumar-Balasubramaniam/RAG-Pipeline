"""
Streamlit UI for the local RAG pipeline.

Run with:
    streamlit run streamlit_app.py

Architecture note — Streamlit re-runs this ENTIRE script on every user
interaction (click, text input, etc.). Anything that must survive across
re-runs (the loaded pipeline, chat history, which document is active)
lives in st.session_state, a dict that persists for the browser session.
Heavy objects (the pipeline, models) are created once and kept there;
without that, every click would rebuild everything from scratch.

The UI has two regions:
    * Sidebar  — document management: upload, library list, delete,
                 Ollama status.
    * Main     — the chat interface for the currently-active document.
"""

import streamlit as st

from config import settings
from src.logger import logger
from src.rag_pipeline import RagPipeline
from src.vector_store import delete_document, list_documents


# ------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Local RAG — Document Q&A",
    page_icon="📄",
    layout="wide",
)


# ------------------------------------------------------------
# Session state initialisation
# Runs once per browser session; survives script re-runs.
# ------------------------------------------------------------
def _init_session_state() -> None:
    """Create the persistent objects if they don't exist yet."""
    if "pipeline" not in st.session_state:
        # One RagPipeline for the whole session. Building it is cheap;
        # it loads models lazily on first real use.
        st.session_state.pipeline = RagPipeline()
        logger.info("Streamlit: created session RagPipeline")

    if "messages" not in st.session_state:
        # Chat history: list of {"role": "user"|"assistant", "content": str}
        st.session_state.messages = []

    if "active_doc_id" not in st.session_state:
        st.session_state.active_doc_id = None

    if "active_doc_name" not in st.session_state:
        st.session_state.active_doc_name = None


# ------------------------------------------------------------
# Ollama health check (shown in the sidebar)
# ------------------------------------------------------------
def _ollama_status() -> tuple[bool, str]:
    """
    Check whether the local Ollama server is reachable and has the model.

    Returns:
        (ok, message) — ok=True if the configured model is available.
    """
    try:
        import ollama
        models = ollama.list().get("models", [])
        names = [m.get("model", m.get("name", "")) for m in models]
        if any(settings.llm_model in n for n in names):
            return True, f"Ollama OK · {settings.llm_model}"
        return False, (
            f"Ollama running but '{settings.llm_model}' not pulled. "
            f"Run: ollama pull {settings.llm_model}"
        )
    except Exception as exc:
        return False, f"Ollama unreachable: {exc}"


# ------------------------------------------------------------
# Sidebar: upload + library + delete + status
# ------------------------------------------------------------
def _render_sidebar() -> None:
    """Render the document-management sidebar."""
    with st.sidebar:
        st.header("📤 Upload a PDF")

        uploaded = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            help="Text or scanned PDFs both work (scanned ones use OCR).",
        )

        if uploaded is not None:
            # A file is staged. Show an explicit button so we don't
            # re-ingest on every script re-run.
            if st.button("📥 Ingest this document", use_container_width=True):
                _handle_upload(uploaded)

        st.divider()
        st.header("📚 Your Documents")

        docs = list_documents()
        if not docs:
            st.caption("No documents yet. Upload one above.")
        else:
            for doc in docs:
                _render_doc_row(doc)

        st.divider()

        # Ollama status indicator
        ok, msg = _ollama_status()
        if ok:
            st.success(msg, icon="✅")
        else:
            st.error(msg, icon="⚠️")


def _render_doc_row(doc: dict) -> None:
    """Render one document row: select button + delete button."""
    doc_id = doc["doc_id"]
    name = doc.get("original_filename", "unknown")
    source = doc.get("source", "?")
    chunks = doc.get("num_chunks", "?")

    is_active = st.session_state.active_doc_id == doc_id
    label = f"{'🟢 ' if is_active else ''}{name}"

    col_select, col_delete = st.columns([4, 1])

    with col_select:
        if st.button(
            label,
            key=f"select_{doc_id}",
            use_container_width=True,
            help=f"{source} · {chunks} chunks",
        ):
            _handle_select(doc_id)

    with col_delete:
        if st.button("🗑", key=f"delete_{doc_id}", help="Delete this document"):
            _handle_delete(doc_id, name)


# ------------------------------------------------------------
# Event handlers
# ------------------------------------------------------------
def _handle_upload(uploaded_file) -> None:
    """Ingest an uploaded PDF and make it the active document."""
    pdf_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    with st.spinner(f"Processing '{filename}' — this can take a while on CPU…"):
        try:
            result = st.session_state.pipeline.ingest(pdf_bytes, filename)
        except Exception as exc:
            logger.error("Streamlit ingest failed | file={} | error={}",
                         filename, exc)
            st.error(f"Could not process '{filename}': {exc}")
            return

    st.session_state.active_doc_id = result.doc_id
    st.session_state.active_doc_name = result.original_filename
    st.session_state.messages = []  # fresh chat for the new document

    if result.was_cached:
        st.success(
            f"'{result.original_filename}' was already indexed — "
            f"loaded instantly from cache ({result.num_chunks} chunks)."
        )
    else:
        st.success(
            f"Indexed '{result.original_filename}' "
            f"({result.source}, {result.num_chunks} chunks)."
        )
    st.rerun()


def _handle_select(doc_id: str) -> None:
    """Load a previously-indexed document as the active one."""
    with st.spinner("Loading document from cache…"):
        try:
            result = st.session_state.pipeline.load_existing(doc_id)
        except Exception as exc:
            logger.error("Streamlit load_existing failed | doc_id={} | "
                         "error={}", doc_id, exc)
            st.error(f"Could not load document: {exc}")
            return

    st.session_state.active_doc_id = result.doc_id
    st.session_state.active_doc_name = result.original_filename
    st.session_state.messages = []
    st.rerun()


def _handle_delete(doc_id: str, name: str) -> None:
    """Delete a document's index + registry entry."""
    try:
        delete_document(doc_id)
    except Exception as exc:
        logger.error("Streamlit delete failed | doc_id={} | error={}",
                     doc_id, exc)
        st.error(f"Could not delete '{name}': {exc}")
        return

    # If we deleted the active document, clear the chat.
    if st.session_state.active_doc_id == doc_id:
        st.session_state.active_doc_id = None
        st.session_state.active_doc_name = None
        st.session_state.messages = []

    st.success(f"Deleted '{name}'.")
    st.rerun()


# ------------------------------------------------------------
# Main area: chat
# ------------------------------------------------------------
def _render_chat() -> None:
    """Render the chat interface for the active document."""
    if st.session_state.active_doc_id is None:
        st.title("📄 Local RAG — Document Q&A")
        st.info(
            "👈 Upload a PDF or pick one from your library to start chatting. "
            "Everything runs locally — your documents never leave this machine."
        )
        return

    st.title(f"💬 {st.session_state.active_doc_name}")
    st.caption(
        "Answers are generated by your local Mistral 7B from this document "
        "only. Nothing is sent to any external service."
    )

    # Replay the chat history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (pinned to the bottom by Streamlit).
    question = st.chat_input("Ask a question about this document…")
    if question:
        _handle_question(question)


def _handle_question(question: str) -> None:
    """Run a question through the pipeline and append the exchange."""
    # Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer.
    with st.chat_message("assistant"):
        with st.spinner("Thinking (local Mistral on CPU — please wait)…"):
            try:
                answer = st.session_state.pipeline.ask(question)
            except Exception as exc:
                logger.error("Streamlit ask failed | error={}", exc)
                answer = f"⚠️ Something went wrong: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main() -> None:
    """App entry point — called once per script re-run."""
    _init_session_state()
    _render_sidebar()
    _render_chat()


main()