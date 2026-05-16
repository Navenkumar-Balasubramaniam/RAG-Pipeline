"""
The RAG pipeline orchestrator.

This is the conductor. The seven modules built so far each do one job;
this class wires them into the two flows the application actually needs:

    ingest(pdf_bytes, filename)  -> index a document (once, then cached)
    ask(question)                -> answer a question about the loaded doc

Design:
    * A RagPipeline instance is bound to ONE document at a time. You
      ingest a document (or load an already-indexed one), then ask
      questions against it. To switch documents, ingest another.
    * Ingesting an already-indexed document is near-instant: it skips
      straight to loading the persisted index (no re-chunk/re-embed).
      This is the payoff of the Step 6 persistence work.
    * The retriever is built once per loaded document and reused across
      questions (building it re-reads nodes; we don't want that per query).

The Streamlit app (Step 9) holds one RagPipeline in session state and
calls these two methods. Nothing in the UI needs to know the internals.
"""

from dataclasses import dataclass
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from src.chunker import chunk_text
from src.llm import answer_with_context, rewrite_query
from src.logger import logger
from src.pdf_loader import load_pdf
from src.reranker import get_reranker, rerank
from src.retriever import build_hybrid_retriever, retrieve_chunks
from src.vector_store import (
    build_and_persist_index,
    compute_doc_id,
    is_indexed,
    load_index,
    load_registry,
)


@dataclass
class IngestResult:
    """
    Outcome of ingesting a document.

    Attributes:
        doc_id:            The content-hash id of the document.
        original_filename: Display name of the file.
        source:            "text" or "ocr" — how the PDF was processed.
        num_chunks:        Number of chunks the document produced.
        was_cached:        True if we loaded an existing index instead of
                           building a new one (i.e. this doc was already
                           ingested before — no re-embedding happened).
    """
    doc_id: str
    original_filename: str
    source: str
    num_chunks: int
    was_cached: bool


class RagPipeline:
    """
    Orchestrates document ingestion and question answering.

    One instance is bound to one document at a time. Typical lifecycle:

        pipeline = RagPipeline()
        result = pipeline.ingest(pdf_bytes, "report.pdf")
        answer = pipeline.ask("What were the Q3 results?")
        answer = pipeline.ask("And Q4?")          # reuses the retriever
        pipeline.ingest(other_bytes, "other.pdf") # switch documents
    """

    def __init__(self) -> None:
        # State for the currently-loaded document. All None until ingest().
        self._doc_id: Optional[str] = None
        self._index: Optional[VectorStoreIndex] = None
        self._nodes: Optional[list[BaseNode]] = None
        self._retriever = None  # QueryFusionRetriever, built lazily
        logger.debug("RagPipeline instantiated (no document loaded yet)")

    # --------------------------------------------------------
    # Flow A: ingest
    # --------------------------------------------------------
    def ingest(self, pdf_bytes: bytes, original_filename: str) -> IngestResult:
        """
        Ingest a PDF: index it if new, or load its existing index if seen.

        Args:
            pdf_bytes:         Raw bytes of the uploaded PDF.
            original_filename: Display name (shown in the UI / registry).

        Returns:
            An IngestResult describing what happened (including whether
            the index was served from cache).

        Raises:
            PDFLoadError:  If the PDF can't be opened.
            ValueError:    If the document produces no chunks.
        """
        doc_id = compute_doc_id(pdf_bytes)
        logger.info(
            "Ingest requested | file={} | doc_id={}",
            original_filename, doc_id,
        )

        # --- Fast path: already indexed -> just load it ---
        if is_indexed(doc_id):
            logger.info("Document already indexed; loading from disk | "
                        "doc_id={}", doc_id)
            index = load_index(doc_id)
            registry = load_registry()
            meta = registry.get(doc_id, {})

            self._doc_id = doc_id
            self._index = index
            # We need the nodes for BM25. Pull them out of the loaded
            # index's docstore rather than re-chunking the PDF.
            self._nodes = list(index.docstore.docs.values())
            self._retriever = None  # rebuilt lazily on first ask()

            return IngestResult(
                doc_id=doc_id,
                original_filename=meta.get("original_filename",
                                           original_filename),
                source=meta.get("source", "unknown"),
                num_chunks=meta.get("num_chunks", len(self._nodes)),
                was_cached=True,
            )

        # --- Slow path: new document -> full pipeline ---
        logger.info("New document; running full ingest pipeline | "
                    "doc_id={}", doc_id)

        # We need the bytes on disk for pdf_loader (it works with paths).
        # Write to the uploads dir under the doc_id so it's traceable.
        from config import settings
        settings.ensure_directories()
        pdf_path = settings.uploads_dir / f"{doc_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        text, source = load_pdf(pdf_path)
        nodes = chunk_text(text)

        if not nodes:
            raise ValueError(
                f"Document {original_filename!r} produced no chunks "
                f"(empty or unreadable after extraction/OCR)."
            )

        meta = build_and_persist_index(
            doc_id=doc_id,
            original_filename=original_filename,
            nodes=nodes,
            source=source,
        )

        # Load the index we just built so it's ready for questions.
        self._doc_id = doc_id
        self._index = load_index(doc_id)
        self._nodes = nodes
        self._retriever = None

        return IngestResult(
            doc_id=doc_id,
            original_filename=original_filename,
            source=source,
            num_chunks=meta["num_chunks"],
            was_cached=False,
        )

    # --------------------------------------------------------
    # Load an already-indexed document by id (for the UI's
    # "pick from previously uploaded" feature)
    # --------------------------------------------------------
    def load_existing(self, doc_id: str) -> IngestResult:
        """
        Bind this pipeline to an already-indexed document by its id.

        Used by the Streamlit sidebar when the user picks a document
        from the registry instead of uploading a new file.

        Args:
            doc_id: The id of a document known to be in the registry.

        Returns:
            An IngestResult (was_cached is always True here).

        Raises:
            FileNotFoundError: If no persisted index exists for doc_id.
        """
        logger.info("Loading existing document | doc_id={}", doc_id)
        index = load_index(doc_id)
        registry = load_registry()
        meta = registry.get(doc_id, {})

        self._doc_id = doc_id
        self._index = index
        self._nodes = list(index.docstore.docs.values())
        self._retriever = None

        return IngestResult(
            doc_id=doc_id,
            original_filename=meta.get("original_filename", "unknown"),
            source=meta.get("source", "unknown"),
            num_chunks=meta.get("num_chunks", len(self._nodes)),
            was_cached=True,
        )

    # --------------------------------------------------------
    # Flow B: ask
    # --------------------------------------------------------
    def ask(self, question: str) -> str:
        """
        Answer a question about the currently-loaded document.

        Pipeline: rewrite -> hybrid retrieve -> rerank -> generate.

        Args:
            question: The user's natural-language question.

        Returns:
            The grounded answer string from local Mistral.

        Raises:
            RuntimeError: If no document has been ingested/loaded yet.
            LLMError:     If the local LLM is unreachable.
        """
        if self._index is None or self._nodes is None:
            raise RuntimeError(
                "No document loaded. Call ingest() or load_existing() first."
            )

        logger.info("Question received | doc_id={} | q={!r}",
                    self._doc_id, question)

        # Build the hybrid retriever once per loaded document, then reuse.
        if self._retriever is None:
            self._retriever = build_hybrid_retriever(self._index, self._nodes)

        # Step 1: optionally rewrite the query (config-gated, in llm.py).
        search_query = rewrite_query(question)

        # Step 2: hybrid retrieval (vector + BM25).
        retrieved = retrieve_chunks(self._retriever, search_query)

        # Step 3: rerank and keep the top-N.
        reranked = rerank(get_reranker(), search_query, retrieved)

        # Step 4: generate the grounded answer with local Mistral.
        context_chunks = [n.get_content() for n in reranked]
        answer = answer_with_context(question, context_chunks)

        logger.info("Answer produced | doc_id={} | answer_chars={}",
                    self._doc_id, len(answer))
        return answer

    # --------------------------------------------------------
    # Introspection helper for the UI
    # --------------------------------------------------------
    @property
    def loaded_doc_id(self) -> Optional[str]:
        """The id of the currently-loaded document, or None."""
        return self._doc_id