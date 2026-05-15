"""
Tests for src/retriever.py and src/reranker.py.

Retrieval needs a real embedding model + a built index; reranking needs
the real cross-encoder. So the end-to-end tests are @slow. One fast test
covers the empty-input guard (pure logic, no model).
"""

import pytest


# ============================================================
# Fast: pure-logic guard
# ============================================================

class TestRerankGuards:

    def test_rerank_empty_nodes_returns_empty(self, mocker):
        """rerank() on an empty list must return [] without touching a model."""
        from src.reranker import rerank

        # Pass a dummy reranker object; it must never be used because the
        # empty-list guard returns before any model call.
        dummy = mocker.MagicMock()
        out = rerank(dummy, "any query", [])
        assert out == []
        dummy.postprocess_nodes.assert_not_called()


# ============================================================
# Slow: full retrieval + rerank against a real tiny index
# ============================================================

@pytest.mark.slow
class TestHybridRetrievalAndRerank:

    @pytest.fixture
    def tiny_index_and_nodes(self):
        """
        Build a small in-memory index from a few sentences covering
        distinct topics, plus the nodes (BM25 needs the raw nodes).
        """
        from llama_index.core import VectorStoreIndex
        from src.chunker import chunk_text
        from src.embedder import get_embed_model

        text = (
            "The Eiffel Tower is located in Paris, France. "
            "It was completed in 1889 for the World's Fair. "
            "Photosynthesis converts sunlight into chemical energy in plants. "
            "Chlorophyll in chloroplasts absorbs the light. "
            "The error code TX-409 indicates a payment gateway timeout. "
            "Retrying the transaction usually resolves TX-409."
        )
        nodes = chunk_text(text)
        index = VectorStoreIndex(nodes, embed_model=get_embed_model())
        return index, nodes

    def test_vector_strength_semantic_match(self, tiny_index_and_nodes):
        """
        A paraphrased query with no shared keywords should still retrieve
        the relevant chunk — that's the dense/vector retriever working.
        """
        from src.retriever import build_hybrid_retriever, retrieve_chunks

        index, nodes = tiny_index_and_nodes
        retriever = build_hybrid_retriever(index, nodes)

        results = retrieve_chunks(retriever, "Where is the famous iron tower?")
        combined = " ".join(r.get_content() for r in results).lower()
        assert "eiffel" in combined or "paris" in combined

    def test_bm25_strength_exact_token(self, tiny_index_and_nodes):
        """
        An exact code lookup should retrieve the chunk containing it
        verbatim — that's the sparse/BM25 retriever working.
        """
        from src.retriever import build_hybrid_retriever, retrieve_chunks

        index, nodes = tiny_index_and_nodes
        retriever = build_hybrid_retriever(index, nodes)

        results = retrieve_chunks(retriever, "TX-409")
        combined = " ".join(r.get_content() for r in results)
        assert "TX-409" in combined

    def test_rerank_orders_most_relevant_first(self, tiny_index_and_nodes):
        """
        After reranking, the top chunk for a photosynthesis question
        should be the photosynthesis chunk.
        """
        from src.retriever import build_hybrid_retriever, retrieve_chunks
        from src.reranker import get_reranker, rerank

        index, nodes = tiny_index_and_nodes
        retriever = build_hybrid_retriever(index, nodes)

        query = "How do plants make energy from light?"
        retrieved = retrieve_chunks(retriever, query)
        reranked = rerank(get_reranker(), query, retrieved)

        assert len(reranked) >= 1
        top_text = reranked[0].get_content().lower()
        assert "photosynthesis" in top_text or "chlorophyll" in top_text