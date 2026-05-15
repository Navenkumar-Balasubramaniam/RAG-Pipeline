"""
PDF text extraction with OCR fallback.

This module is the entry point for converting an arbitrary PDF into plain
text suitable for chunking and embedding. It handles two cases transparently:

1. **Text-based PDFs** (the common case): created by Word, LaTeX, browsers
   exporting to PDF, etc. The text is stored as actual text data inside the
   file. We extract it directly with PyMuPDF — fast, perfect fidelity.

2. **Image-based PDFs** (scanned documents, photographed pages): the file
   contains only images of pages, no text. PyMuPDF will return an empty or
   near-empty string. We detect this and fall back to OCR (Step 4B).

The detection is a simple heuristic: extract via PyMuPDF first; if the
result is below `settings.pdf_min_text_length` characters, treat it as
image-based and OCR. The threshold (default 50 chars) is large enough to
ignore a stray header or page number that PyMuPDF might catch on a scanned
PDF, but small enough that a very short real document still flows through.

Public API:

    text, source = load_pdf("path/to/document.pdf")

Where `source` is one of: "text" or "ocr" — useful for logging and for the
Streamlit UI to show the user how the document was processed.
"""

from pathlib import Path

import fitz  # PyMuPDF

from config import settings
from src.logger import logger


# ------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------
class PDFLoadError(Exception):
    """Raised when a PDF cannot be opened or processed at all."""


# ------------------------------------------------------------
# Text extraction via PyMuPDF
# ------------------------------------------------------------
def extract_text_with_pymupdf(pdf_path: Path) -> str:
    """
    Extract text from every page of a PDF using PyMuPDF (fitz).

    For text-based PDFs this returns the actual stored text. For image-based
    PDFs it returns an empty string or near-empty (maybe a few stray
    characters from compression metadata).

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Concatenated text from all pages, separated by newlines.
        Empty string if the PDF has no extractable text.

    Raises:
        PDFLoadError: If the file cannot be opened (corrupt, encrypted,
            not actually a PDF, etc.).
    """
    logger.debug("Extracting text via PyMuPDF | path={}", pdf_path)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        # PyMuPDF raises a generic Exception, so we wrap it in our own
        # type to give callers a clean way to handle "this PDF is broken"
        logger.error("Failed to open PDF | path={} | error={}", pdf_path, exc)
        raise PDFLoadError(f"Could not open PDF: {pdf_path}") from exc

    page_texts: list[str] = []
    try:
        for page_idx, page in enumerate(doc):
            page_text = page.get_text()
            page_texts.append(page_text)
            logger.debug(
                "Extracted page | page={}/{} | chars={}",
                page_idx + 1, doc.page_count, len(page_text),
            )
    finally:
        # Always close the document, even if iteration raised
        doc.close()

    full_text = "\n".join(page_texts)
    logger.info(
        "PyMuPDF extraction done | path={} | pages={} | total_chars={}",
        pdf_path, len(page_texts), len(full_text),
    )
    return full_text


# ------------------------------------------------------------
# The orchestrator: detection + dispatch
# ------------------------------------------------------------
def load_pdf(pdf_path: str | Path) -> tuple[str, str]:
    """
    Load a PDF and return its text, falling back to OCR if needed.

    Decision logic:
        - Try PyMuPDF text extraction first (fast, lossless for text PDFs).
        - If the extracted text is shorter than settings.pdf_min_text_length,
          treat the PDF as image-based and run OCR (slow, lossy, but works
          on scanned documents).

    Args:
        pdf_path: Path to a PDF file. Can be str or Path.

    Returns:
        A tuple of (text, source) where:
            text:   The extracted text (possibly empty if OCR also fails).
            source: "text" if PyMuPDF succeeded, "ocr" if we fell back.

    Raises:
        PDFLoadError: If the file can't be opened at all.
        FileNotFoundError: If the path doesn't exist.
    """
    # Normalise the input to a Path object for consistent handling.
    # We accept str OR Path so callers don't need to convert.
    path = Path(pdf_path)

    if not path.exists():
        logger.error("PDF not found | path={}", path)
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF | path={}", path)

    # --- Step 1: try direct text extraction ---
    text = extract_text_with_pymupdf(path)

    # --- Step 2: decide based on length ---
    if len(text.strip()) >= settings.pdf_min_text_length:
        logger.info(
            "Detected text-based PDF | path={} | chars={}",
            path, len(text),
        )
        return text, "text"

    # --- Step 3: fall back to OCR ---
    # (Implementation of run_ocr() is added in Step 4B.)
    logger.warning(
        "Text extraction yielded only {} chars (< {} threshold). "
        "Falling back to OCR | path={}",
        len(text.strip()), settings.pdf_min_text_length, path,
    )
    text = run_ocr(path)
    return text, "ocr"


# ------------------------------------------------------------
# OCR pipeline — placeholder, implemented in Step 4B
# ------------------------------------------------------------
def run_ocr(pdf_path: Path) -> str:
    """
    Run the OCR pipeline on an image-based PDF.

    This is a placeholder; the real implementation lands in Step 4B and will:
      1. Render each PDF page to an image
      2. Preprocess each image with OpenCV (grayscale, threshold, denoise, resize)
      3. Run PaddleOCR on each cleaned image
      4. Concatenate the per-page text

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Extracted text.

    Raises:
        NotImplementedError: Always, until Step 4B.
    """
    raise NotImplementedError("OCR pipeline lands in Step 4B")