"""
Semantic chunking of document text.

Splitting strategy: SemanticSplitterNodeParser from LlamaIndex.

Instead of cutting text at fixed character counts (which slices sentences
and arguments in half), the semantic splitter:

    1. Splits the text into sentences.
    2. Embeds each sentence.
    3. Walks through consecutive sentences measuring how different
       neighbouring sentences are (cosine distance between embeddings).
    4. Inserts a chunk boundary wherever that difference spikes above
       a percentile threshold — i.e. where the topic visibly shifts.

The result: every chunk is about roughly one coherent idea, which makes
retrieval far more precise than fixed-size chunking.

Two tunables (from config.py):
    chunk_buffer_size:          sentences of context considered either side
                                of a candidate break (1 = compare adjacent
                                sentences; higher = smoother, fewer breaks).
    chunk_breakpoint_percentile: 0-99. Higher = fewer, larger chunks. 95
                                means "only break at the top-5% biggest
                                topic shifts".

Public API:

    nodes = chunk_text("the full document text...")
    # nodes is a list of LlamaIndex TextNode objects ready for indexing
"""

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode

from config import settings
from src.embedder import get_embed_model
from src.logger import logger


def chunk_text(text: str) -> list[BaseNode]:
    """
    Split raw document text into semantically coherent nodes (chunks).

    Args:
        text: The full plain-text content of a document (from pdf_loader).

    Returns:
        A list of LlamaIndex nodes. Each node holds one chunk of text plus
        metadata. Returns an empty list if the input text is empty/blank.

    Notes:
        The semantic splitter needs an embedding model to measure sentence
        similarity, so this function pulls the shared embedder. That model
        is reused later for indexing — no duplicate loading.
    """
    if not text or not text.strip():
        logger.warning("chunk_text called with empty text; returning no nodes")
        return []

    logger.info("Chunking text | input_chars={}", len(text))

    # LlamaIndex works in terms of Documents; wrap the raw string in one.
    document = Document(text=text)

    # Build the semantic splitter with our configured thresholds.
    splitter = SemanticSplitterNodeParser(
        buffer_size=settings.chunk_buffer_size,
        breakpoint_percentile_threshold=settings.chunk_breakpoint_percentile,
        embed_model=get_embed_model(),
    )

    nodes = splitter.get_nodes_from_documents([document])

    logger.info(
        "Chunking done | input_chars={} | chunks={}",
        len(text), len(nodes),
    )
    for i, node in enumerate(nodes):
        logger.debug(
            "Chunk | idx={} | chars={}", i, len(node.get_content())
        )

    return nodes