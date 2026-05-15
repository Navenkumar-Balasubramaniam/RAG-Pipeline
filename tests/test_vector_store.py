"""
Tests for src/vector_store.py.

Most tests here are FAST: hashing, registry read/write, delete logic,
and existence checks need no models. Only the build+persist+load
round-trip needs the real embedding model, so that one is @slow.

Critical technique: we redirect settings paths to a pytest tmp_path so
tests never touch your real data/indices folder. This is done via a
fixture that monkeypatches the settings object.
"""

from pathlib import Path

import pytest


# ------------------------------------------------------------
# Isolate every test from the real data directory
# ------------------------------------------------------------
@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    """
    Point settings.indices_dir and registry_path at a throwaway temp dir.

    Without this, tests would read/write your real data/indices folder
    and could clobber real indexed documents. monkeypatch automatically
    undoes the change when the test finishes.

    Yields:
        The temp indices directory Path (for assertions).
    """
    from config import settings

    indices_dir = tmp_path / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "indices_dir", indices_dir)
    monkeypatch.setattr(settings, "registry_path", indices_dir / "registry.json")

    yield indices_dir


# ============================================================
# Hashing — fast, pure
# ============================================================

class TestComputeDocId:

    def test_same_bytes_same_id(self):
        from src.vector_store import compute_doc_id
        a = compute_doc_id(b"hello world pdf bytes")
        b = compute_doc_id(b"hello world pdf bytes")
        assert a == b

    def test_different_bytes_different_id(self):
        from src.vector_store import compute_doc_id
        a = compute_doc_id(b"document one")
        b = compute_doc_id(b"document two")
        assert a != b

    def test_id_is_16_hex_chars(self):
        from src.vector_store import compute_doc_id
        doc_id = compute_doc_id(b"anything")
        assert len(doc_id) == 16
        # All chars must be valid hex
        int(doc_id, 16)  # raises ValueError if not hex


# ============================================================
# Registry — fast
# ============================================================

class TestRegistry:

    def test_empty_when_no_file(self, isolated_storage):
        from src.vector_store import load_registry
        assert load_registry() == {}

    def test_save_and_reload_roundtrip(self, isolated_storage):
        from src.vector_store import load_registry, _save_registry
        data = {"abc123": {"doc_id": "abc123", "original_filename": "x.pdf"}}
        _save_registry(data)
        assert load_registry() == data

    def test_corrupt_registry_returns_empty(self, isolated_storage):
        """A garbage registry file should degrade gracefully to empty."""
        from config import settings
        from src.vector_store import load_registry

        settings.registry_path.write_text("{ this is not valid json")
        assert load_registry() == {}


# ============================================================
# is_indexed / delete / list — fast
# ============================================================

class TestIsIndexedAndDelete:

    def test_not_indexed_when_absent(self, isolated_storage):
        from src.vector_store import is_indexed
        assert is_indexed("doesnotexist") is False

    def test_delete_nonexistent_is_noop(self, isolated_storage):
        """Deleting something that isn't there must not raise."""
        from src.vector_store import delete_document
        delete_document("ghost")  # should simply log + return

    def test_list_documents_empty_initially(self, isolated_storage):
        from src.vector_store import list_documents
        assert list_documents() == []

    def test_list_documents_sorted_newest_first(self, isolated_storage):
        from src.vector_store import _save_registry, list_documents
        _save_registry({
            "old": {"doc_id": "old", "indexed_at": "2026-01-01T00:00:00"},
            "new": {"doc_id": "new", "indexed_at": "2026-05-01T00:00:00"},
        })
        docs = list_documents()
        assert [d["doc_id"] for d in docs] == ["new", "old"]


# ============================================================
# Full round-trip — SLOW (needs the real embedding model)
# ============================================================

@pytest.mark.slow
class TestBuildPersistLoadRoundtrip:

    def test_build_persist_load_and_delete(self, isolated_storage):
        """
        End-to-end: chunk -> build+persist -> is_indexed -> load -> delete.
        This proves persistence actually works across a fresh load.
        """
        from src.chunker import chunk_text
        from src.vector_store import (
            build_and_persist_index,
            delete_document,
            is_indexed,
            load_index,
        )

        text = (
            "Renewable energy adoption is accelerating worldwide. "
            "Solar panel costs have fallen dramatically over the past decade. "
            "Wind turbines now generate power at competitive prices. "
            "Battery storage technology continues to improve rapidly."
        )
        nodes = chunk_text(text)
        assert len(nodes) >= 1

        doc_id = "testdoc00000001"

        # Build + persist
        meta = build_and_persist_index(doc_id, "energy.pdf", nodes, "text")
        assert meta["doc_id"] == doc_id
        assert meta["num_chunks"] == len(nodes)

        # Should now report as indexed
        assert is_indexed(doc_id) is True

        # Load it back — must not raise, must be queryable
        index = load_index(doc_id)
        retriever = index.as_retriever(similarity_top_k=2)
        results = retriever.retrieve("How cheap is solar power?")
        assert len(results) >= 1

        # Delete and confirm it's gone
        delete_document(doc_id)
        assert is_indexed(doc_id) is False

    def test_build_with_no_nodes_raises(self, isolated_storage):
        from src.vector_store import build_and_persist_index
        with pytest.raises(ValueError, match="no nodes"):
            build_and_persist_index("x", "empty.pdf", [], "text")