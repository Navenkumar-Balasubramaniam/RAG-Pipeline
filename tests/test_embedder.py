"""
Tests for src/embedder.py.

The embedding model load is slow (~3-5s) and downloads ~130 MB on first
run, so the tests that actually load it are marked @pytest.mark.slow.
"""

import pytest


@pytest.mark.slow
class TestEmbedder:
    """Tests that exercise the real embedding model."""

    def test_embed_text_returns_float_list(self):
        """embed_text should return a non-empty list of floats."""
        from src.embedder import embed_text

        vector = embed_text("The quick brown fox jumps over the lazy dog.")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(x, float) for x in vector)

    def test_embedding_dimension_is_384(self):
        """bge-small-en-v1.5 produces 384-dimensional vectors."""
        from src.embedder import embed_text

        vector = embed_text("hello world")
        assert len(vector) == 384

    def test_similar_texts_have_closer_vectors(self):
        """
        Sanity check that embeddings capture meaning: two related
        sentences should be closer than two unrelated ones.
        """
        import numpy as np
        from src.embedder import embed_text

        def cosine(a, b):
            a, b = np.array(a), np.array(b)
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        cat1 = embed_text("The cat sat on the mat.")
        cat2 = embed_text("A feline rested on the rug.")
        finance = embed_text("Quarterly revenue exceeded forecasts.")

        sim_related = cosine(cat1, cat2)
        sim_unrelated = cosine(cat1, finance)

        # Related sentences must be more similar than unrelated ones
        assert sim_related > sim_unrelated

    def test_singleton_returns_same_instance(self):
        """get_embed_model must return the same object every call."""
        from src.embedder import get_embed_model

        a = get_embed_model()
        b = get_embed_model()
        assert a is b  # identity, not just equality


@pytest.mark.slow
class TestChunker:
    """Tests that exercise the real semantic splitter (needs embeddings)."""

    def test_empty_text_returns_no_nodes(self):
        """Empty / whitespace text should yield an empty list, no crash."""
        from src.chunker import chunk_text

        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_chunks_a_multi_topic_document(self):
        """
        A document with clearly distinct topics should produce at least
        one node, and the nodes' combined content should preserve the
        source text's key phrases.
        """
        from src.chunker import chunk_text

        text = (
            "Photosynthesis is the process by which plants convert light "
            "into chemical energy. Chlorophyll absorbs sunlight in the "
            "chloroplasts. "
            "Meanwhile, the stock market reacts to interest rate changes. "
            "Investors rebalance portfolios when the central bank shifts "
            "monetary policy. "
            "Separately, volcanic eruptions release sulfur dioxide into "
            "the stratosphere, which can cool global temperatures."
        )

        nodes = chunk_text(text)

        assert len(nodes) >= 1
        combined = " ".join(n.get_content() for n in nodes)
        assert "Photosynthesis" in combined
        assert "stock market" in combined
        assert "volcanic" in combined