"""
Central configuration for the RAG pipeline.

This module defines a single `Settings` class that holds every tunable
parameter and path used anywhere in the project. Values are loaded
(in order of precedence) from:

    1. Environment variables prefixed with RAG_
    2. The .env file in the project root
    3. The defaults declared in this file

Importing pattern (used everywhere in the codebase):

    from config import settings
    print(settings.llm_model)

Why a single Settings object instead of scattered constants:

    * One source of truth for every tunable
    * Type-validated at startup (catches bad values immediately)
    * Secrets live in .env, never in code
    * Easy to override per-environment without code changes
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ------------------------------------------------------------
# Project root: resolved once, used to anchor every other path
# Path(__file__) is this config.py file; .resolve() makes it absolute;
# .parent gives the directory containing it (the project root)
# ------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env.

    Every field declared here can be overridden by setting an environment
    variable with the prefix `RAG_` (e.g. RAG_LLM_MODEL=qwen2.5:7b will
    override the `llm_model` default below).

    Field names use snake_case; the corresponding env var is the upper-cased
    name prefixed with RAG_. Example: `top_k_retrieval` -> `RAG_TOP_K_RETRIEVAL`.
    """

    # --------------------------------------------------------
    # Pydantic Settings configuration
    # --------------------------------------------------------
    model_config = SettingsConfigDict(
        env_prefix="RAG_",                  # All env vars must start with RAG_
        env_file=PROJECT_ROOT / ".env",     # Load from .env in the project root
        env_file_encoding="utf-8",
        case_sensitive=False,               # RAG_LLM_MODEL == rag_llm_model
        extra="ignore",                     # Ignore unknown env vars (don't crash)
    )

    # --------------------------------------------------------
    # Filesystem paths
    # All paths anchored to PROJECT_ROOT, so the project is portable
    # --------------------------------------------------------
    project_root: Path = Field(
        default=PROJECT_ROOT,
        description="Absolute path to the project root directory.",
    )
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Root folder for all persisted data (uploads + indices).",
    )
    uploads_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "uploads",
        description="Where user-uploaded PDFs are stored.",
    )
    indices_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "indices",
        description="Where persisted vector indices are stored (one folder per document).",
    )
    registry_path: Path = Field(
        default=PROJECT_ROOT / "data" / "indices" / "registry.json",
        description="JSON file listing all indexed documents and their metadata.",
    )
    logs_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Where rotating log files are written.",
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )

    # --------------------------------------------------------
    # PDF extraction
    # --------------------------------------------------------
    pdf_min_text_length: int = Field(
        default=50,
        ge=0,
        description=(
            "Minimum number of characters extracted by PyMuPDF to consider a PDF "
            "as text-based. Below this, we fall back to OCR. 50 chars is enough to "
            "rule out 'this PDF had only a single header word that PyMuPDF caught'."
        ),
    )
    ocr_languages: list[str] = Field(
        default=["en"],
        description="PaddleOCR language codes to load (e.g. ['en'], ['en', 'fr']).",
    )

    # --------------------------------------------------------
    # OCR image preprocessing (used only on image-based PDFs)
    # --------------------------------------------------------
    ocr_resize_dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description=(
            "Resolution to render PDF pages at before OCR. 300 DPI is the sweet "
            "spot — higher means more accuracy but slower OCR."
        ),
    )
    ocr_adaptive_threshold_block_size: int = Field(
        default=11,
        ge=3,
        description=(
            "OpenCV adaptive threshold block size (must be odd). Larger = "
            "smoother thresholding, smaller = more sensitive to local contrast."
        ),
    )
    ocr_bilateral_filter_diameter: int = Field(
        default=9,
        ge=1,
        description="Diameter of pixel neighborhood for OpenCV bilateral filter.",
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------
    chunk_buffer_size: int = Field(
        default=1,
        ge=1,
        description=(
            "Semantic splitter buffer: how many sentences on either side of a "
            "potential break to consider when deciding to split."
        ),
    )
    chunk_breakpoint_percentile: int = Field(
        default=95,
        ge=50,
        le=99,
        description=(
            "Semantic splitter percentile threshold. Higher = fewer, larger chunks. "
            "95 means: split only at the top-5% most-dissimilar sentence boundaries."
        ),
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description=(
            "HuggingFace model id for embeddings. bge-small is 33M params, fast on "
            "CPU, and competitive with much larger models on retrieval tasks."
        ),
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------
    top_k_retrieval: int = Field(
        default=5,
        ge=1,
        le=50,
        description="How many chunks each retriever returns (vector and BM25 each).",
    )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model used for reranking retrieved chunks.",
    )
    reranker_device: str = Field(
        default="cpu",
        description=(
            "Compute device for the reranker cross-encoder. 'cpu' for "
            "CPU-only machines (this project's default). Set to 'cuda' "
            "if running on an NVIDIA GPU. Must be an explicit string — "
            "the sbert-rerank library's None default fails its own "
            "schema validation."
        ),
    )
    top_n_rerank: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "How many chunks survive reranking and reach the LLM. Should be "
            "<= top_k_retrieval, since the reranker can only filter, not add."
        ),
    )

    # --------------------------------------------------------
    # Local LLM (Ollama)
    # --------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="HTTP URL of the local Ollama server.",
    )
    llm_model: str = Field(
        default="mistral:7b-instruct",
        description=(
            "Ollama model identifier. Swap this to switch models — e.g. "
            "'qwen2.5:7b' or 'llama3.1:8b'. Remember to `ollama pull` first."
        ),
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature. Low = deterministic, high = creative.",
    )
    llm_max_tokens: int = Field(
        default=700,
        ge=50,
        le=4096,
        description="Maximum tokens generated by the LLM in one response.",
    )
    enable_query_rewriting: bool = Field(
        default=True,
        description=(
            "If True, rewrite the user query via LLM before retrieval (improves "
            "recall, costs one extra LLM call). On CPU, disabling this roughly "
            "halves response latency."
        ),
    )

    # --------------------------------------------------------
    # Validators — run automatically when settings are loaded
    # --------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Ensure log_level is one of the canonical Python logging levels."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got: {v!r}")
        return v_upper

    @field_validator("ocr_adaptive_threshold_block_size")
    @classmethod
    def _validate_block_size_is_odd(cls, v: int) -> int:
        """OpenCV requires the adaptive threshold block size to be odd."""
        if v % 2 == 0:
            raise ValueError(
                f"ocr_adaptive_threshold_block_size must be odd (OpenCV requirement), "
                f"got: {v}"
            )
        return v

    def ensure_directories(self) -> None:
        """
        Create all required directories if they don't exist.

        Called once at application startup (from main.py / streamlit_app.py)
        so we don't crash later trying to write into a missing folder.
        """
        for path in (self.data_dir, self.uploads_dir, self.indices_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Module-level singleton
# Import this from anywhere: `from config import settings`
# Pydantic instantiates it once, validation runs immediately, fail-fast.
# ------------------------------------------------------------
settings = Settings()