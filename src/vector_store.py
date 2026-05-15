"""
Vector index persistence and the document registry.

This module owns everything about *storing* and *retrieving* indexed
documents on disk, plus the registry that catalogues them. It is the
feature that lets the app "remember" documents across restarts instead
of re-embedding every time.

Two responsibilities:

1. **Per-document index storage.** Each document is chunked + embedded
   once, then its LlamaIndex VectorStoreIndex is persisted to its own
   folder under settings.indices_dir. The folder name is the document's
   content hash (doc_id), so re-uploading an identical file is detected
   and skipped.

2. **The registry.** A single JSON file (settings.registry_path) mapping
   doc_id -> metadata (original filename, chunk count, when indexed,
   how it was processed). This is the catalogue the UI reads to list
   available documents and the source of truth for delete operations.

Public API:

    doc_id = compute_doc_id(pdf_bytes)
    registry = load_registry()
    is_indexed(doc_id) -> bool
    build_and_persist_index(doc_id, filename, nodes, source) -> metadata
    load_index(doc_id) -> VectorStoreIndex
    delete_document(doc_id) -> None
    list_documents() -> list[metadata]
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import BaseNode

from config import settings
from src.embedder import get_embed_model
from src.logger import logger


# ------------------------------------------------------------
# Document identity
# ------------------------------------------------------------
def compute_doc_id(pdf_bytes: bytes) -> str:
    """
    Compute a stable document ID from raw PDF file bytes.

    We hash the file *content* (not its name), so:
      * the same file always maps to the same id (enables dedupe)
      * different files never collide, even with identical filenames
      * the id is always a safe folder name (hex chars only)

    Args:
        pdf_bytes: The raw bytes of the PDF file.

    Returns:
        A 16-character hex string (truncated SHA-256). 16 hex chars =
        64 bits of entropy — astronomically collision-safe for the
        scale of a personal/document-library app.
    """
    full_hash = hashlib.sha256(pdf_bytes).hexdigest()
    doc_id = full_hash[:16]
    logger.debug("Computed doc_id | doc_id={}", doc_id)
    return doc_id


# ------------------------------------------------------------
# Registry read / write
# ------------------------------------------------------------
def load_registry() -> dict[str, dict]:
    """
    Load the document registry from disk.

    Returns:
        A dict mapping doc_id -> metadata dict. Empty dict if no
        registry file exists yet (first run).
    """
    if not settings.registry_path.exists():
        logger.debug("No registry file yet | path={}", settings.registry_path)
        return {}

    try:
        with open(settings.registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        logger.debug("Loaded registry | entries={}", len(registry))
        return registry
    except (json.JSONDecodeError, OSError) as exc:
        # A corrupt registry shouldn't crash the whole app. Log loudly
        # and start from empty — the index folders still exist on disk
        # and could be rebuilt if needed.
        logger.error("Registry unreadable, treating as empty | error={}", exc)
        return {}


def _save_registry(registry: dict[str, dict]) -> None:
    """
    Write the registry back to disk atomically.

    We write to a temp file then rename, so a crash mid-write can't
    leave a half-written (corrupt) registry. rename() is atomic on
    POSIX filesystems.

    Args:
        registry: The full registry dict to persist.
    """
    settings.ensure_directories()
    tmp_path = settings.registry_path.with_suffix(".json.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    tmp_path.replace(settings.registry_path)
    logger.debug("Saved registry | entries={}", len(registry))


# ------------------------------------------------------------
# Existence check
# ------------------------------------------------------------
def is_indexed(doc_id: str) -> bool:
    """
    Return True if a document is already indexed and present on disk.

    Checks both the registry entry AND that the index folder actually
    exists — guards against the registry and disk drifting out of sync.

    Args:
        doc_id: The document id from compute_doc_id().

    Returns:
        True if the document can be loaded without re-indexing.
    """
    registry = load_registry()
    if doc_id not in registry:
        return False

    index_path = settings.indices_dir / doc_id
    exists = index_path.exists() and any(index_path.iterdir())
    if not exists:
        logger.warning(
            "Registry has {} but index folder missing/empty | path={}",
            doc_id, index_path,
        )
    return exists


# ------------------------------------------------------------
# Build + persist
# ------------------------------------------------------------
def build_and_persist_index(
    doc_id: str,
    original_filename: str,
    nodes: list[BaseNode],
    source: str,
) -> dict:
    """
    Build a vector index from chunked nodes and persist it to disk.

    Args:
        doc_id:            Content hash from compute_doc_id().
        original_filename: The file's display name (for the UI).
        nodes:             Chunked nodes from chunker.chunk_text().
        source:            "text" or "ocr" (how the PDF was processed).

    Returns:
        The metadata dict that was written to the registry for this doc.

    Raises:
        ValueError: If nodes is empty (nothing to index).
    """
    if not nodes:
        raise ValueError(
            f"Cannot build index for {original_filename!r}: no nodes "
            f"(document produced no chunks)."
        )

    index_path = settings.indices_dir / doc_id
    logger.info(
        "Building index | doc_id={} | file={} | chunks={}",
        doc_id, original_filename, len(nodes),
    )

    # Build the vector index. The embed model is our shared singleton,
    # so this reuses already-loaded weights.
    index = VectorStoreIndex(
        nodes,
        embed_model=get_embed_model(),
        show_progress=False,
    )

    # Persist all of LlamaIndex's storage (vectors, docstore, index meta)
    # into the per-document folder.
    index_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(index_path))
    logger.info("Persisted index to disk | path={}", index_path)

    # Update the registry.
    metadata = {
        "doc_id": doc_id,
        "original_filename": original_filename,
        "source": source,
        "num_chunks": len(nodes),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path),
    }
    registry = load_registry()
    registry[doc_id] = metadata
    _save_registry(registry)

    logger.info("Document indexed and registered | doc_id={}", doc_id)
    return metadata


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
def load_index(doc_id: str) -> VectorStoreIndex:
    """
    Load a previously-persisted index from disk.

    This is the payoff of persistence: no re-chunking, no re-embedding.
    LlamaIndex reads the stored vectors straight back into an index.

    Args:
        doc_id: The document id to load.

    Returns:
        A ready-to-query VectorStoreIndex.

    Raises:
        FileNotFoundError: If no persisted index exists for this doc_id.
    """
    index_path = settings.indices_dir / doc_id

    if not index_path.exists():
        raise FileNotFoundError(
            f"No persisted index for doc_id={doc_id} at {index_path}"
        )

    logger.info("Loading index from disk | doc_id={}", doc_id)
    storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
    index = load_index_from_storage(
        storage_context,
        embed_model=get_embed_model(),
    )
    logger.info("Index loaded | doc_id={}", doc_id)
    return index


# ------------------------------------------------------------
# Delete
# ------------------------------------------------------------
def delete_document(doc_id: str) -> None:
    """
    Permanently remove a document's index folder and registry entry.

    Args:
        doc_id: The document id to delete.

    Notes:
        Idempotent: deleting something that doesn't exist is a no-op
        with a warning, not an error.
    """
    registry = load_registry()
    index_path = settings.indices_dir / doc_id

    if doc_id not in registry and not index_path.exists():
        logger.warning("delete_document: nothing to delete | doc_id={}", doc_id)
        return

    # Remove the index folder (and everything in it).
    if index_path.exists():
        shutil.rmtree(index_path)
        logger.info("Deleted index folder | path={}", index_path)

    # Remove the registry entry.
    if doc_id in registry:
        del registry[doc_id]
        _save_registry(registry)
        logger.info("Removed registry entry | doc_id={}", doc_id)


# ------------------------------------------------------------
# List
# ------------------------------------------------------------
def list_documents() -> list[dict]:
    """
    Return metadata for every indexed document, newest first.

    This is what the Streamlit sidebar calls to render the document
    library.

    Returns:
        A list of metadata dicts, sorted by indexed_at descending.
    """
    registry = load_registry()
    docs = list(registry.values())
    docs.sort(key=lambda d: d.get("indexed_at", ""), reverse=True)
    logger.debug("Listed documents | count={}", len(docs))
    return docs