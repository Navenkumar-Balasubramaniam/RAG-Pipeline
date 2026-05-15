"""
Cross-encoder reranking of retrieved chunks.

Retrieval (vector + BM25) is fast but scores the query and each chunk
*independently* — it never looks at them together. A cross-encoder does:
it takes (query, chunk) as a single input and outputs a precise relevance
score. This is far more accurate but too slow to run over a whole corpus,
so we only run it over the handful of candidates retrieval already found,
then keep the best `settings.top_n_rerank`.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (from config). Small, CPU-
friendly, trained specifically for passage reranking.

The reranker is a LlamaIndex "node postprocessor": it takes the list of
NodeWithScore from retrieval and returns a shorter, reordered list.

Public API:

    reranker = get_reranker()
    top_nodes = rerank(reranker, query, retrieved_nodes)
"""

from typing import Optional

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from config import settings
from src.logger import logger


# ------------------------------------------------------------
# Singleton — the cross-encoder is ~80 MB, load once per process
# ------------------------------------------------------------
_reranker: Optional[SentenceTransformerRerank] = None


def get_reranker() -> SentenceTransformerRerank:
    """
    Return the shared cross-encoder reranker, loading it on first call.

    Returns:
        A configured SentenceTransformerRerank postprocessor that keeps
        the top settings.top_n_rerank nodes.
    """
    global _reranker

    if _reranker is None:
        logger.info(
            "Loading reranker | model={} | top_n={}",
            settings.reranker_model, settings.top_n_rerank,
        )
        _reranker = SentenceTransformerRerank(
            model=settings.reranker_model,
            top_n=settings.top_n_rerank,
            device=settings.reranker_device,        # Explicit: this project is CPU-only
                                 # (no GPU). Also required because this
                                 # version of sbert-rerank defaults
                                 # device to None, which fails its own
                                 # Pydantic str validation.
        )
        logger.info("Reranker ready")

    return _reranker


def rerank(
    reranker: SentenceTransformerRerank,
    query: str,
    nodes: list[NodeWithScore],
) -> list[NodeWithScore]:
    """
    Re-score and trim retrieved nodes by true query relevance.

    Args:
        reranker: The postprocessor from get_reranker().
        query:    The user question (same one used for retrieval).
        nodes:    The NodeWithScore list from the hybrid retriever.

    Returns:
        A reordered, shortened list — at most settings.top_n_rerank nodes,
        best first. Empty input yields empty output (no crash).
    """
    if not nodes:
        logger.warning("rerank called with no nodes; returning empty")
        return []

    logger.info("Reranking | n_in={} | keeping_top={}",
                len(nodes), settings.top_n_rerank)

    # The postprocessor API takes a QueryBundle, not a raw string.
    reranked = reranker.postprocess_nodes(
        nodes,
        query_bundle=QueryBundle(query_str=query),
    )

    logger.info("Reranked | n_out={}", len(reranked))
    for i, r in enumerate(reranked):
        logger.debug(
            "Reranked result | rank={} | score={:.4f}",
            i, r.score or 0.0,
        )
    return reranked