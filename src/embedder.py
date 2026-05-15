"""
Text embedding using a local HuggingFace model.

An "embedding" is a fixed-length vector of floats that captures the
*meaning* of a piece of text. Texts with similar meaning have vectors
that are close together (by cosine similarity); unrelated texts have
distant vectors. This is the mathematical foundation of semantic search.

We use BAAI/bge-small-en-v1.5:
    * 33M parameters — small enough to run fast on CPU
    * 384-dimensional output vectors
    * Competitive with much larger models on retrieval benchmarks
    * Runs fully locally — no data leaves the machine (matches the
      project's privacy goal)

The model is wrapped as a singleton: loading it takes a few seconds and
~150 MB RAM, so we do it once per process and reuse it everywhere
(chunking, indexing, and query-time all use the same instance).

Public API:

    embed_model = get_embed_model()      # the LlamaIndex embedding object
    vector = embed_text("some text")     # a single 384-float list
"""

from typing import Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import settings
from src.logger import logger


# ------------------------------------------------------------
# Singleton — load the embedding model once per process
# ------------------------------------------------------------
_embed_model: Optional[HuggingFaceEmbedding] = None


def get_embed_model() -> HuggingFaceEmbedding:
    """
    Return the shared HuggingFace embedding model, loading it on first call.

    The model id comes from settings.embedding_model (default
    BAAI/bge-small-en-v1.5). On first call the weights are downloaded to
    the HuggingFace cache (~/.cache/huggingface) — about 130 MB — and
    loaded into RAM. Every later call returns the same instance.

    Returns:
        A LlamaIndex HuggingFaceEmbedding wrapping the configured model.
    """
    global _embed_model

    if _embed_model is None:
        logger.info(
            "Loading embedding model | model={}", settings.embedding_model
        )
        _embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
        )
        logger.info("Embedding model ready | model={}", settings.embedding_model)

    return _embed_model


def embed_text(text: str) -> list[float]:
    """
    Embed a single string into its vector representation.

    This is mostly a convenience wrapper for tests and ad-hoc use. The
    actual pipeline embeds chunks in bulk via the vector index, which is
    far more efficient than calling this in a loop.

    Args:
        text: The text to embed.

    Returns:
        A list of floats (length 384 for bge-small) representing the
        text's position in semantic space.
    """
    model = get_embed_model()
    vector = model.get_text_embedding(text)
    logger.debug("Embedded text | chars={} | dim={}", len(text), len(vector))
    return vector