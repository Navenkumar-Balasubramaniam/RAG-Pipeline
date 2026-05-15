"""
Tests for src/llm.py.

Strategy:
    * FAST tests mock ollama.chat — they verify OUR logic (message
      construction, system prompts, error wrapping, fail-soft rewrite)
      without loading or calling the real 4 GB model.
    * SLOW tests make one real call to confirm end-to-end connectivity.

The `mocker` fixture comes from pytest-mock (in our environment.yaml).
"""

import pytest


# ============================================================
# generate() — fast, mocked
# ============================================================

class TestGenerate:

    def test_builds_user_message(self, mocker):
        """generate() must send the prompt as a user-role message."""
        fake = mocker.patch(
            "src.llm.ollama.chat",
            return_value={"message": {"content": "hi there"}},
        )
        from src.llm import generate

        out = generate("hello")

        assert out == "hi there"
        # Inspect what we sent to ollama.chat
        _, kwargs = fake.call_args
        messages = kwargs["messages"]
        assert messages[-1] == {"role": "user", "content": "hello"}

    def test_includes_system_message_when_given(self, mocker):
        fake = mocker.patch(
            "src.llm.ollama.chat",
            return_value={"message": {"content": "ok"}},
        )
        from src.llm import generate

        generate("q", system="be terse")

        _, kwargs = fake.call_args
        messages = kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be terse"}
        assert messages[1]["role"] == "user"

    def test_wraps_errors_in_llmerror(self, mocker):
        """A raw ollama failure must surface as our LLMError."""
        mocker.patch(
            "src.llm.ollama.chat",
            side_effect=ConnectionError("connection refused"),
        )
        from src.llm import generate, LLMError

        with pytest.raises(LLMError, match="Ollama service"):
            generate("anything")


# ============================================================
# rewrite_query() — fast, mocked
# ============================================================

class TestRewriteQuery:

    def test_returns_original_when_disabled(self, mocker, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "enable_query_rewriting", False)

        from src.llm import rewrite_query
        assert rewrite_query("what is x?") == "what is x?"

    def test_returns_rewritten_when_enabled(self, mocker, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "enable_query_rewriting", True)
        mocker.patch(
            "src.llm.ollama.chat",
            return_value={"message": {"content": "expanded query terms"}},
        )
        from src.llm import rewrite_query
        assert rewrite_query("x?") == "expanded query terms"

    def test_fails_soft_to_original_on_error(self, mocker, monkeypatch):
        """If the rewrite call errors, we must fall back to the original."""
        from config import settings
        monkeypatch.setattr(settings, "enable_query_rewriting", True)
        mocker.patch(
            "src.llm.ollama.chat",
            side_effect=ConnectionError("down"),
        )
        from src.llm import rewrite_query
        assert rewrite_query("original question") == "original question"


# ============================================================
# answer_with_context() — fast, mocked
# ============================================================

class TestAnswerWithContext:

    def test_context_chunks_embedded_in_prompt(self, mocker):
        fake = mocker.patch(
            "src.llm.ollama.chat",
            return_value={"message": {"content": "the answer"}},
        )
        from src.llm import answer_with_context

        out = answer_with_context(
            "What colour is the sky?",
            ["The sky appears blue due to Rayleigh scattering."],
        )

        assert out == "the answer"
        _, kwargs = fake.call_args
        user_msg = kwargs["messages"][-1]["content"]
        # The context and question must both be in the prompt
        assert "Rayleigh scattering" in user_msg
        assert "What colour is the sky?" in user_msg


# ============================================================
# Real Ollama connectivity — SLOW
# ============================================================

@pytest.mark.slow
class TestRealOllama:

    def test_real_generate_returns_text(self):
        """One real call to confirm Ollama + Mistral are reachable."""
        from src.llm import generate
        out = generate("Reply with the single word: pong")
        assert isinstance(out, str)
        assert len(out) > 0