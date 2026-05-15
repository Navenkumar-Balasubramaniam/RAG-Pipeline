"""
Hybrid retrieval: dense (vector) + sparse (BM25) fusion.

Given a loaded VectorStoreIndex and a query, this returns the most
relevant chunks by combining two complementary retrieval strategies:

1. **Vector retrieval** — embeds the query and finds chunks with similar
   embeddings. Strong at *meaning*: "reduce latency" matches a chunk
   about "improving response times" even with no shared words. Weak at
   exact tokens (codes, names, acronyms).

2. **BM25 retrieval** — classic keyword scoring (term frequency / inverse
   document frequency). Strong at *exact matches*: an error code like
   "TX-409" is found verbatim. Weak at synonyms / paraphrase.

LlamaIndex's QueryFusionRetriever runs both and fuses the ranked lists
(reciprocal rank fusion), giving us precision AND recall. This mirrors
the hybrid approach from the original notebook.

Public API:

    retriever = build_hybrid_retriever(index, nodes)
    results = retriever.retrieve("the user's question")
"""

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from config import settings
from src.logger import logger

# ------------------------------------------------------------
# A no-op LLM for LlamaIndex components that demand one.
#
# LlamaIndex silently defaults to OpenAI in many components and
# crashes with "No API key found for OpenAI" if none is set.
# In 0.12.x, passing llm=None does NOT help — the Settings.llm
# property getter re-resolves the OpenAI default anyway. The only
# reliable fix is to hand the component a real no-op LLM object.
#
# MockLLM is built into LlamaIndex for exactly this. It is never
# actually called here (num_queries=1 means no query generation),
# it just satisfies the "an LLM object must exist" requirement.
# The ONLY real LLM in this project remains our local Mistral,
# accessed exclusively through src/llm.py.
# ------------------------------------------------------------
_NOOP_LLM = MockLLM()


def build_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
) -> QueryFusionRetriever:
    """
    Build a fused vector + BM25 retriever.

    Args:
        index: A loaded VectorStoreIndex (from vector_store.load_index or
            freshly built). Provides the dense/vector retriever.
        nodes: The chunk nodes. BM25 needs the raw nodes to build its
            keyword index (it doesn't use embeddings, so it can't read
            them out of the vector store — it indexes the text itself).

    Returns:
        A QueryFusionRetriever ready for .retrieve(query).

    Notes:
        We deliberately do NOT use an LLM inside the fusion retriever
        (num_queries=1, no query generation) — query rewriting is handled
        explicitly in our llm module so the behaviour is in one place and
        config-controlled. This keeps retrieval deterministic and fast.
    """
    # A retriever can never return more items than exist in the corpus.
    # Clamp the configured top-k to the number of available nodes so that
    # short documents (fewer chunks than top_k_retrieval) don't crash
    # BM25, which strictly rejects k > corpus_size. This is a real
    # production concern: a one-paragraph PDF could chunk into 1-2 nodes.
    effective_top_k = min(settings.top_k_retrieval, len(nodes))

    logger.info(
        "Building hybrid retriever | nodes={} | configured_top_k={} | "
        "effective_top_k={}",
        len(nodes), settings.top_k_retrieval, effective_top_k,
    )

    # Dense/vector retriever from the index.
    vector_retriever = index.as_retriever(
        similarity_top_k=effective_top_k,
    )

    # Sparse/BM25 retriever built directly from the nodes' text.
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=effective_top_k,
    )

    # Fuse the two. mode="reciprocal_rerank" = reciprocal rank fusion:
    # a robust, parameter-free way to combine two ranked lists. We set
    # num_queries=1 so it does NOT call an LLM to generate query variants
    # (we handle rewriting ourselves, deterministically).
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=settings.top_k_retrieval,
        num_queries=1,
        mode="reciprocal_rerank",
        llm=_NOOP_LLM,
        use_async=False,
        verbose=False,
    )

    logger.info("Hybrid retriever ready")
    return fusion_retriever


def retrieve_chunks(
    retriever: QueryFusionRetriever,
    query: str,
) -> list[NodeWithScore]:
    """
    Run a retrieval query and return scored nodes.

    Thin wrapper that adds logging around .retrieve(). Kept separate so
    the pipeline orchestrator (Step 8) has a clean, logged entry point
    and tests have something easy to assert on.

    Args:
        retriever: The retriever from build_hybrid_retriever().
        query:     The (possibly rewritten) user question.

    Returns:
        A list of NodeWithScore, best first. Length up to top_k_retrieval.
    """
    logger.info("Retrieving | query={!r}", query)
    results = retriever.retrieve(query)
    logger.info("Retrieved | n_results={}", len(results))
    for i, r in enumerate(results):
        logger.debug(
            "Result | rank={} | score={:.4f} | chars={}",
            i, r.score or 0.0, len(r.get_content()),
        )
    return results