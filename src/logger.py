"""
Central logging setup for the RAG pipeline.

This module configures Loguru once at import-time and exposes a single
`logger` object that every other module in the project imports and uses:

    from src.logger import logger
    logger.info("Loaded document with %d pages", n)
    logger.error("OCR failed on page %d", page_idx)

The same logger instance is shared across all modules — Loguru handles this
automatically. There's no per-module setup; just import and use.

Output goes to two places:

1. **Console** (stderr) — colourised, human-readable, controlled by the
   RAG_LOG_LEVEL env var (default INFO). Useful during development.

2. **Rotating files** in logs/ — every day a new file is created, kept for
   7 days, then automatically deleted. Always logs at DEBUG level so the
   file contains everything regardless of the console level. This is your
   safety net: "what happened last Tuesday at 3 PM?"

Log levels reminder:

    DEBUG    Detailed diagnostic info ("entering function X with args Y")
    INFO     Normal progress ("loaded 47 chunks from document")
    WARNING  Something unexpected but recoverable ("OCR confidence low")
    ERROR    Something failed ("could not extract text from PDF")
    CRITICAL Unrecoverable failure ("Ollama server unreachable, exiting")
"""

import sys

from loguru import logger

from config import settings


def _configure_logger() -> None:
    """
    Configure Loguru's global logger with our console + file sinks.

    Called once at module import time. Safe to call again (it removes
    existing handlers first so we don't get duplicate output).

    A 'sink' in Loguru = a destination for log records (e.g. stderr, a file).
    We attach two sinks: one for the console, one for the rotating file.
    """
    # Make sure the logs directory exists before we point a file sink at it
    settings.ensure_directories()

    # Wipe any existing handlers (including Loguru's default stderr sink).
    # Without this, calling configure twice would duplicate every log line.
    logger.remove()

    # ----------------------------------------------------------
    # Sink 1: console output (stderr), colourised
    # Level controlled by settings.log_level (env-overridable)
    # ----------------------------------------------------------
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "| <level>{level: <8}</level> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,   # On exceptions, show the full call stack
        diagnose=True,    # On exceptions, show variable values at each frame
    )

    # ----------------------------------------------------------
    # Sink 2: rotating file in logs/
    # Always at DEBUG level (the file is our complete safety-net record)
    # Rotates daily; keeps 7 days; old files are deleted automatically
    # ----------------------------------------------------------
    logger.add(
        settings.logs_dir / "rag_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} "
            "| {level: <8} "
            "| {name}:{function}:{line} "
            "- {message}"
        ),
        rotation="00:00",         # New file at midnight every day
        retention="7 days",       # Delete files older than 7 days
        compression="zip",        # Compress rotated files to save space
        enqueue=True,             # Thread-safe writes (matters for Streamlit)
        backtrace=True,
        diagnose=True,
    )

    logger.debug("Logger configured | console_level=%s | logs_dir=%s",
                 settings.log_level, settings.logs_dir)


# ------------------------------------------------------------
# Configure once when this module is first imported
# ------------------------------------------------------------
_configure_logger()


# Re-export for clean imports:  from src.logger import logger
__all__ = ["logger"]