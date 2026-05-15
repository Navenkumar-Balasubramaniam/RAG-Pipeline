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
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np

from config import settings
from src.logger import logger

# ------------------------------------------------------------
# PaddleOCR singleton
# Initialised lazily on first use; reused across all subsequent calls.
# Loading PaddleOCR is expensive (~5-10s + first-time model download),
# so we never want to do it more than once per process.
# ------------------------------------------------------------
_paddle_ocr: Optional["object"] = None


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
# PaddleOCR lazy initialisation
# ------------------------------------------------------------
def _get_paddle_ocr():
    """
    Return the shared PaddleOCR instance, creating it on first call.

    PaddleOCR's first instantiation:
        * Downloads ~25 MB of detection + recognition models to ~/.paddleocr/
        * Loads them into RAM (~500 MB resident)
        * Takes 5-10 seconds

    All subsequent calls reuse the same loaded instance, so per-page OCR
    runs in roughly 1-3 seconds rather than 10+. Without this singleton
    pattern, batch OCR on a multi-page PDF would be unusably slow.

    Returns:
        A configured PaddleOCR instance ready to process images.
    """
    global _paddle_ocr

    if _paddle_ocr is None:
        logger.info("Initialising PaddleOCR | langs={}", settings.ocr_languages)
        # Local import: paddleocr loads slowly and triggers paddlepaddle
        # initialisation, so we don't want to pay that cost until OCR is
        # actually needed. Keeping it inside the function defers the load.
        from paddleocr import PaddleOCR

        # use_angle_cls=True enables auto-rotation of text lines, which
        # helps with scanned documents that aren't perfectly aligned.
        # show_log=False quiets PaddleOCR's noisy startup messages.
        _paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang=settings.ocr_languages[0],  # PaddleOCR 2.x takes a single lang
            show_log=False,
        )
        logger.info("PaddleOCR ready")

    return _paddle_ocr


# ------------------------------------------------------------
# Image preprocessing (deterministic, easy to unit-test)
# ------------------------------------------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Clean up a raw page image to maximise OCR accuracy.

    Pipeline (matches the description in your project requirements):

        1. Grayscale conversion — OCR doesn't care about colour, and
           single-channel images are smaller / faster to process.
        2. Bilateral filter — denoise while preserving edges. Unlike a
           gaussian blur (which smears letter edges), bilateral filtering
           smooths flat regions but keeps the sharp transitions of text.
        3. Adaptive threshold — convert grayscale to clean binary
           (black text on white background). Adaptive means each pixel's
           threshold is based on its neighbourhood, which handles
           uneven lighting in scanned documents.

    Note on ordering: we denoise BEFORE thresholding. Doing it the other
    way wipes out subtle pixel gradations that the filter needs to
    distinguish noise from real edges.

    Resizing for DPI uplift happens at render time (`fitz.Pixmap`),
    not here — by the time we receive an image, it's already at our
    target DPI from `settings.ocr_resize_dpi`.

    Args:
        image: An H×W×3 BGR colour image (as returned by OpenCV or
            fitz.Pixmap → np.ndarray conversion).

    Returns:
        An H×W single-channel binary image suitable for OCR.
    """
    logger.debug("Preprocessing image | shape={}", image.shape)

    # Step 1: grayscale.
    # cv2.cvtColor with COLOR_BGR2GRAY: standard formula
    # 0.299*R + 0.587*G + 0.114*B (matches human luminance perception).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: bilateral filter for edge-preserving denoising.
    # Args: source, diameter, sigmaColor, sigmaSpace.
    # Larger sigma values = more smoothing. The diameter comes from config.
    denoised = cv2.bilateralFilter(
        gray,
        d=settings.ocr_bilateral_filter_diameter,
        sigmaColor=75,
        sigmaSpace=75,
    )

    # Step 3: adaptive threshold to binarise.
    # ADAPTIVE_THRESH_GAUSSIAN_C — neighbour pixels are gaussian-weighted.
    # THRESH_BINARY — output is 0 or 255.
    # blockSize — size of the neighbourhood used to calculate threshold.
    # C — constant subtracted from the weighted mean (fine-tune).
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=settings.ocr_adaptive_threshold_block_size,
        C=2,
    )

    logger.debug("Preprocessed image | shape={}", binary.shape)
    return binary


# ------------------------------------------------------------
# Render a PDF page to a numpy image at target DPI
# ------------------------------------------------------------
def _render_page_to_image(page: fitz.Page, dpi: int) -> np.ndarray:
    """
    Convert a single PDF page to an OpenCV-compatible BGR image array.

    PyMuPDF returns pages as Pixmap objects. We:
        1. Set a transform matrix that scales the page to our target DPI
           (PDF default is 72 DPI; we want 300 for crisp OCR).
        2. Extract the raw pixel buffer as RGB.
        3. Convert to a numpy array and swap to BGR (OpenCV's native format).

    Args:
        page: A fitz.Page object from an open document.
        dpi:  Target render resolution. 300 is a good default; higher
              improves OCR on small text but slows everything down.

    Returns:
        H×W×3 BGR uint8 numpy array.
    """
    # 72 is the PDF coordinate baseline (1 point = 1/72 inch).
    # Scaling by dpi/72 yields the requested DPI.
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Pixmap → numpy. The buffer is RGB (h × w × 3); we reshape and convert.
    img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )

    # OpenCV uses BGR by default; convert from PyMuPDF's RGB.
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------
# The OCR orchestrator
# ------------------------------------------------------------
def run_ocr(pdf_path: Path) -> str:
    """
    Run the full OCR pipeline on an image-based PDF.

    For each page:
        1. Render the page to an image at `settings.ocr_resize_dpi`
        2. Preprocess the image (grayscale + denoise + threshold)
        3. Send the cleaned image through PaddleOCR
        4. Extract the recognised text strings

    Per-page text is joined with newlines to form the final document text.

    Args:
        pdf_path: Path to a PDF.

    Returns:
        Extracted text from all pages, joined by newlines. May be empty
        if OCR found no readable text.

    Raises:
        PDFLoadError: If the PDF cannot be opened.
    """
    logger.info("Starting OCR pipeline | path={} | dpi={}",
                pdf_path, settings.ocr_resize_dpi)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF for OCR | path={} | error={}",
                     pdf_path, exc)
        raise PDFLoadError(f"Could not open PDF: {pdf_path}") from exc

    ocr = _get_paddle_ocr()
    page_texts: list[str] = []

    try:
        for page_idx, page in enumerate(doc):
            logger.debug("OCR page | page={}/{}", page_idx + 1, doc.page_count)

            # Render and preprocess
            raw_image = _render_page_to_image(page, settings.ocr_resize_dpi)
            cleaned = preprocess_image(raw_image)

            # PaddleOCR returns a nested structure:
            #     [ [ [bbox, (text, confidence)], ... ] ]
            # The outer list is per-image; we always pass one image so we
            # take result[0]. Each inner item is a detected text line.
            result = ocr.ocr(cleaned, cls=True)

            page_text_parts: list[str] = []
            if result and result[0]:
                for line in result[0]:
                    # line = [bbox_polygon, (text_string, confidence_float)]
                    if line and len(line) >= 2:
                        text_tuple = line[1]
                        if isinstance(text_tuple, (list, tuple)) and len(text_tuple) >= 1:
                            page_text_parts.append(text_tuple[0])

            page_text = "\n".join(page_text_parts)
            page_texts.append(page_text)
            logger.debug(
                "OCR page done | page={} | chars={}",
                page_idx + 1, len(page_text),
            )

    finally:
        doc.close()

    full_text = "\n".join(page_texts)
    logger.info(
        "OCR pipeline done | path={} | pages={} | total_chars={}",
        pdf_path, len(page_texts), len(full_text),
    )
    return full_text