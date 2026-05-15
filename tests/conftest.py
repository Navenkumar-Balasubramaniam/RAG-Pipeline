"""
Shared pytest fixtures for the entire test suite.

Pytest auto-discovers anything in conftest.py and makes it available to
every test file as a function parameter — no imports needed. Each fixture
defined below is a reusable piece of test setup.

When a test function takes a fixture by name as a parameter, pytest
runs the fixture function and passes its return value:

    def test_something(sample_text_pdf):
        # sample_text_pdf is the Path returned by the fixture below
        assert sample_text_pdf.exists()

Fixtures with scope="session" run once for the whole test run.
Fixtures with the default scope="function" run once per test.
"""

from pathlib import Path

import fitz  # PyMuPDF
import pytest


# ------------------------------------------------------------
# Fixtures directory
# ------------------------------------------------------------
# All generated test PDFs live here. We use session scope so the PDFs
# are created once per pytest run, not once per test function.
# ------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """The folder where test fixture files live."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def sample_text_pdf(fixtures_dir: Path) -> Path:
    """
    Generate a small text-based PDF for tests that need a 'happy path'
    text-extraction case.

    Created once per test session and reused. The file path is returned
    so tests can pass it directly to the function under test.

    Returns:
        Path to a small PDF containing extractable text.
    """
    pdf_path = fixtures_dir / "sample_text.pdf"

    if not pdf_path.exists():
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            (
                "This is a test PDF.\n\n"
                "It contains real text data that PyMuPDF should "
                "be able to extract directly without OCR.\n\n"
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            ),
            fontsize=12,
        )
        doc.save(pdf_path)
        doc.close()

    return pdf_path


@pytest.fixture(scope="session")
def sample_blank_pdf(fixtures_dir: Path) -> Path:
    """
    Generate a blank PDF for tests that need to verify the OCR-fallback
    branch is taken when text extraction yields nothing.

    Returns:
        Path to a PDF with one blank page (no extractable text).
    """
    pdf_path = fixtures_dir / "sample_blank.pdf"

    if not pdf_path.exists():
        doc = fitz.open()
        doc.new_page()  # blank page, no text inserted
        doc.save(pdf_path)
        doc.close()

    return pdf_path


@pytest.fixture
def nonexistent_pdf(tmp_path: Path) -> Path:
    """
    Return a path that definitely does not exist on disk.

    Uses pytest's built-in `tmp_path` fixture (a unique temp dir for
    each test). The file is never created, so functions under test
    that check existence will correctly fail.

    Returns:
        Path to a file that does not exist.
    """
    return tmp_path / "does_not_exist.pdf"