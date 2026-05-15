"""
Tests for src/pdf_loader.py.

What's covered:
    - extract_text_with_pymupdf: returns text for text PDFs, empty for blanks
    - load_pdf: routes correctly to "text" vs "ocr" based on content
    - load_pdf: error handling (missing files, corrupt files)

What's NOT covered yet:
    - The actual OCR pipeline (implemented in Step 4B, tests added then)

Fixtures used (defined in conftest.py):
    - sample_text_pdf:  small PDF with real text
    - sample_blank_pdf: PDF with one empty page (triggers OCR fallback)
    - nonexistent_pdf:  path that doesn't exist
"""

from pathlib import Path

import pytest

from src.pdf_loader import (
    PDFLoadError,
    extract_text_with_pymupdf,
    load_pdf,
)


# ============================================================
# extract_text_with_pymupdf
# ============================================================

class TestExtractTextWithPymupdf:
    """Direct PyMuPDF extraction (no detection / fallback logic)."""

    def test_extracts_text_from_text_pdf(self, sample_text_pdf: Path):
        """A text-based PDF should yield non-empty text."""
        text = extract_text_with_pymupdf(sample_text_pdf)

        assert isinstance(text, str)
        assert len(text) > 0
        # The fixture content should contain a known phrase
        assert "Lorem ipsum" in text

    def test_returns_empty_string_for_blank_pdf(self, sample_blank_pdf: Path):
        """A PDF with no text should yield an empty/whitespace-only string."""
        text = extract_text_with_pymupdf(sample_blank_pdf)

        assert isinstance(text, str)
        # Blank page may have a stray newline from page join, but no real text
        assert text.strip() == ""

    def test_raises_pdf_load_error_on_corrupt_file(self, tmp_path: Path):
        """Opening a file that isn't a valid PDF should raise PDFLoadError."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("this is not a pdf")

        with pytest.raises(PDFLoadError):
            extract_text_with_pymupdf(fake_pdf)


# ============================================================
# load_pdf — the orchestrator
# ============================================================

class TestLoadPdf:
    """Detection logic: text-PDF vs image-PDF routing."""

    def test_text_pdf_returns_source_text(self, sample_text_pdf: Path):
        """A text-based PDF should be detected and routed to 'text' source."""
        text, source = load_pdf(sample_text_pdf)

        assert source == "text"
        assert "Lorem ipsum" in text

    def test_text_pdf_accepts_string_path(self, sample_text_pdf: Path):
        """load_pdf should accept str paths, not only Path objects."""
        # Pass as a string instead of a Path
        text, source = load_pdf(str(sample_text_pdf))
        assert source == "text"
        assert len(text) > 0

    def test_blank_pdf_triggers_ocr_branch(self, sample_blank_pdf: Path):
        """
        A blank PDF should fall through to OCR. Until Step 4B implements
        OCR, the placeholder raises NotImplementedError. The fact that we
        reach this branch at all proves detection routed correctly.
        """
        with pytest.raises(NotImplementedError, match="OCR pipeline"):
            load_pdf(sample_blank_pdf)

    def test_missing_file_raises_filenotfounderror(self, nonexistent_pdf: Path):
        """Loading a missing file should raise FileNotFoundError, clearly."""
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            load_pdf(nonexistent_pdf)