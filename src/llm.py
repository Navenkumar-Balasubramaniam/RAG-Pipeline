"""
Local LLM access via Ollama (Mistral 7B by default).

This module is the *only* place in the codebase that talks to the
language model. Everything else (retrieval, the pipeline orchestrator,
Streamlit) calls into here. Centralising it means:

    * The model is swappable from config.py alone (settings.llm_model)
    * Prompt construction lives in one auditable place
    * Tests can mock this single module instead of the whole pipeline

Privacy note: Ollama runs entirely on localhost. No prompt, document
chunk, or answer ever leaves the machine. This is the core reason we
chose a local model over a hosted API.

Two public functions:

    generate(prompt, system=...)        -> raw completion for any prompt
    answer_with_context(question, ctx)  -> RAG answer grounded in chunks
    rewrite_query(question)             -> optional query expansion
"""

from typing import Optional

import ollama

from config import settings
from src.logger import logger


# ------------------------------------------------------------
# Low-level: a single call to the model
# ------------------------------------------------------------
def generate(prompt: str, system: Optional[str] = None) -> str:
    """
    Send one prompt to the local LLM and return its text response.

    Args:
        prompt: The user-role content for the model.
        system: Optional system-role instruction that steers behaviour
            (tone, constraints, role). Sent as a separate message.

    Returns:
        The model's response text, stripped of leading/trailing space.

    Raises:
        LLMError: If Ollama is unreachable or returns an error. We wrap
            the underlying exception so callers have one error type to
            handle and the user gets an actionable message ("is Ollama
            running?") rather than a raw connection traceback.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    logger.debug(
        "LLM call | model={} | prompt_chars={} | has_system={}",
        settings.llm_model, len(prompt), system is not None,
    )

    try:
        response = ollama.chat(
            model=settings.llm_model,
            messages=messages,
            options={
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        )
    except Exception as exc:
        logger.error("LLM call failed | model={} | error={}",
                     settings.llm_model, exc)
        raise LLMError(
            f"Could not reach the local LLM ({settings.llm_model}). "
            f"Is the Ollama service running and the model pulled? "
            f"Original error: {exc}"
        ) from exc

    content = response["message"]["content"].strip()
    logger.debug("LLM response | response_chars={}", len(content))
    return content


# ------------------------------------------------------------
# Custom exception
# ------------------------------------------------------------
class LLMError(Exception):
    """Raised when the local LLM is unreachable or errors out."""


# ------------------------------------------------------------
# Query rewriting (optional, config-gated)
# ------------------------------------------------------------
_REWRITE_SYSTEM = (
    "You rewrite a user's question into a single, self-contained search "
    "query that maximises document retrieval recall. Expand abbreviations, "
    "add likely synonyms, and remove conversational filler. Respond with "
    "ONLY the rewritten query — no preamble, no quotes, no explanation."
)


def rewrite_query(question: str) -> str:
    """
    Rewrite a user question into a retrieval-optimised query.

    Controlled by settings.enable_query_rewriting. When disabled, returns
    the original question unchanged (saves one LLM call — meaningful on
    CPU where each call is several seconds).

    Args:
        question: The user's original question.

    Returns:
        The rewritten query, or the original if rewriting is disabled
        or the LLM call fails (we fail soft here — a worse query still
        beats no answer).
    """
    if not settings.enable_query_rewriting:
        logger.debug("Query rewriting disabled; using original question")
        return question

    try:
        rewritten = generate(question, system=_REWRITE_SYSTEM)
        logger.info(
            "Rewrote query | original={!r} | rewritten={!r}",
            question, rewritten,
        )
        return rewritten or question
    except LLMError:
        # Fail soft: if rewriting fails, fall back to the raw question
        # rather than failing the whole request.
        logger.warning("Query rewrite failed; falling back to original")
        return question


# ------------------------------------------------------------
# The RAG answer
# ------------------------------------------------------------
_ANSWER_SYSTEM = (
    "You are a precise assistant that answers questions strictly from the "
    "provided context. If the context does not contain the answer, say so "
    "plainly — do not invent facts. Be concise and cite the relevant detail."
)


def answer_with_context(question: str, context_chunks: list[str]) -> str:
    """
    Produce a grounded answer to a question using retrieved context.

    The context chunks are concatenated and embedded in the prompt with
    explicit instructions to answer only from them. This is the core
    anti-hallucination mechanism of RAG.

    Args:
        question: The user's question (original or rewritten).
        context_chunks: The reranked, most-relevant chunks of document text.

    Returns:
        The model's grounded answer.

    Raises:
        LLMError: If the LLM is unreachable.
    """
    if not context_chunks:
        logger.warning("answer_with_context called with no context chunks")

    # Number the chunks so the model can reference them and so the prompt
    # structure is unambiguous.
    context_block = "\n\n".join(
        f"[Context {i + 1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )

    prompt = (
        f"Answer the question using ONLY the context below.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    logger.info(
        "Generating grounded answer | question_chars={} | n_chunks={}",
        len(question), len(context_chunks),
    )
    return generate(prompt, system=_ANSWER_SYSTEM)