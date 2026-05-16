"""
Tests for src/rag_pipeline.py — the orchestrator.

Strategy:
    * One FAST test: ask() before ingest() must raise (pure guard logic).
    * SLOW end-to-end tests exercise the real flow with isolated storage
      so they never touch your real data/indices folder.

The isolated_storage fixture (same pattern as test_vector_store.py)
redirects settings paths to a tmp dir.
"""

from pathlib import Path

import pytest


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    """Redirect all storage paths to a throwaway temp dir."""
    from config import settings

    indices = tmp_path / "indices"
    uploads = tmp_path / "uploads"
    indices.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "indices_dir", indices)
    monkeypatch.setattr(settings, "uploads_dir", uploads)
    monkeypatch.setattr(settings, "registry_path", indices / "registry.json")
    yield tmp_path


# ============================================================
# Fast: guard logic
# ============================================================

class TestPipelineGuards:

    def test_ask_before_ingest_raises(self):
        """Asking with no document loaded must raise a clear error."""
        from src.rag_pipeline import RagPipeline

        pipeline = RagPipeline()
        with pytest.raises(RuntimeError, match="No document loaded"):
            pipeline.ask("anything?")


# ============================================================
# Slow: full end-to-end
# ============================================================

@pytest.mark.slow
class TestPipelineEndToEnd:

    def _make_pdf_bytes(self, text: str) -> bytes:
        """Build a real text PDF in memory and return its bytes."""
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
        data = doc.tobytes()
        doc.close()
        return data

    def test_ingest_then_ask(self, isolated_storage):
        """Full flow: ingest a PDF, then get a grounded answer."""
        from src.rag_pipeline import RagPipeline

        pdf = self._make_pdf_bytes(
            "The Apollo 11 mission landed the first humans on the Moon "
            "on July 20, 1969. Neil Armstrong was the first to walk on "
            "the lunar surface. Buzz Aldrin followed shortly after. "
            "The command module pilot was Michael Collins, who remained "
            "in lunar orbit."
        )

        pipeline = RagPipeline()
        result = pipeline.ingest(pdf, "apollo.pdf")

        assert result.was_cached is False
        assert result.num_chunks >= 1
        assert result.source == "text"

        answer = pipeline.ask("Who was the first person to walk on the Moon?")
        assert "Armstrong" in answer

    def test_reingest_is_cached(self, isolated_storage):
        """
        Ingesting the same bytes twice: the second time must be served
        from cache (was_cached=True), proving persistence/dedup works.
        """
        from src.rag_pipeline import RagPipeline

        pdf = self._make_pdf_bytes(
            "Photosynthesis converts light energy into chemical energy. "
            "It occurs in the chloroplasts of plant cells. Chlorophyll "
            "is the primary pigment that absorbs sunlight."
        )

        p1 = RagPipeline()
        r1 = p1.ingest(pdf, "photo.pdf")
        assert r1.was_cached is False

        # Brand-new pipeline instance, same bytes -> must hit the cache
        p2 = RagPipeline()
        r2 = p2.ingest(pdf, "photo.pdf")
        assert r2.was_cached is True
        assert r2.doc_id == r1.doc_id

        # And it must still answer correctly from the cached index
        answer = p2.ask("Where does photosynthesis occur?")
        assert "chloroplast" in answer.lower()

    def test_load_existing_by_id(self, isolated_storage):
        """A document can be re-bound by doc_id via load_existing()."""
        from src.rag_pipeline import RagPipeline

        pdf = self._make_pdf_bytes(
            "The Great Wall of China is over 13,000 miles long. "
            "Construction spanned multiple dynasties over centuries. "
            "It was built primarily for military defense."
        )

        p1 = RagPipeline()
        r1 = p1.ingest(pdf, "wall.pdf")

        p2 = RagPipeline()
        r2 = p2.load_existing(r1.doc_id)
        assert r2.doc_id == r1.doc_id
        assert r2.was_cached is True

        answer = p2.ask("How long is the Great Wall?")
        assert "13,000" in answer or "13000" in answer