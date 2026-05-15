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

    

    def test_missing_file_raises_filenotfounderror(self, nonexistent_pdf: Path):
        """Loading a missing file should raise FileNotFoundError, clearly."""
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            load_pdf(nonexistent_pdf)
            
# ============================================================
# preprocess_image — pure OpenCV, fast, deterministic
# ============================================================

class TestPreprocessImage:
    """OpenCV preprocessing tests — fast, no models needed."""

    def test_returns_single_channel(self):
        """Preprocessing a 3-channel BGR image should yield grayscale (2-D)."""
        from src.pdf_loader import preprocess_image
        import numpy as np

        # Build a fake 100×100 BGR image filled with mid-gray pixels
        fake_image = np.full((100, 100, 3), 128, dtype=np.uint8)

        result = preprocess_image(fake_image)

        # Output must be single-channel (2-dim) since we binarised
        assert result.ndim == 2
        assert result.shape == (100, 100)

    def test_returns_binary_pixel_values(self):
        """Adaptive threshold output should contain only 0 and 255."""
        from src.pdf_loader import preprocess_image
        import numpy as np

        # A noisy image so the threshold actually has something to threshold
        rng = np.random.default_rng(seed=42)
        fake_image = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

        result = preprocess_image(fake_image)

        unique_vals = set(np.unique(result).tolist())
        # Binary threshold guarantees only 0 and 255 in the output
        assert unique_vals.issubset({0, 255})

    def test_preserves_dimensions(self):
        """Preprocessing must not change image height or width."""
        from src.pdf_loader import preprocess_image
        import numpy as np

        fake_image = np.full((250, 400, 3), 200, dtype=np.uint8)
        result = preprocess_image(fake_image)
        assert result.shape == (250, 400)




@pytest.mark.slow
class TestRunOcr:
    """Integration tests for the full OCR pipeline."""

    def test_ocr_extracts_text_from_image_pdf(self, sample_image_pdf: Path):
        """
        OCR should recognise the text we burnt into the image. The
        match is fuzzy because OCR isn't perfect — we just look for
        a substring we expect to appear.
        """
        from src.pdf_loader import run_ocr

        text = run_ocr(sample_image_pdf)
        assert isinstance(text, str)
        assert len(text) > 0
        # Case-insensitive substring check — OCR may produce all caps,
        # mixed case, or have minor character errors
        assert "HELLO" in text.upper() or "OCR" in text.upper()

    def test_load_pdf_routes_image_pdf_to_ocr(self, sample_image_pdf: Path):
        """End-to-end: load_pdf on an image PDF returns source='ocr'."""
        from src.pdf_loader import load_pdf

        text, source = load_pdf(sample_image_pdf)
        assert source == "ocr"
        assert len(text) > 0