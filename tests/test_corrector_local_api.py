"""Tests for LocalLLMCorrector and APICorrector non-inference paths."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from scroot.config.corrector import APIConfig, LocalConfig
from scroot.corrector.api import APICorrector
from scroot.corrector.local import LocalLLMCorrector


# ─── LocalLLMCorrector ────────────────────────────────────────────────────────

class TestLocalLLMCorrectorPaths:
    def _make(self, model_id="phi4-mini") -> LocalLLMCorrector:
        return LocalLLMCorrector(LocalConfig(model_id=model_id))

    def test_is_available_false_when_llama_cpp_missing(self):
        c = self._make()
        with patch.dict(sys.modules, {"llama_cpp": None}):
            assert c.is_available is False

    def test_is_available_false_when_model_not_downloaded(self):
        c = self._make()
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with patch("scroot.corrector.local.is_model_downloaded", return_value=False):
                assert c.is_available is False

    def test_is_available_true_when_ready(self):
        c = self._make()
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with patch("scroot.corrector.local.is_model_downloaded", return_value=True):
                assert c.is_available is True

    def test_ensure_loaded_raises_without_llama_cpp(self):
        c = self._make()
        with patch.dict(sys.modules, {"llama_cpp": None}):
            with pytest.raises(RuntimeError, match="llama-cpp-python"):
                c._ensure_loaded()

    def test_ensure_loaded_raises_model_not_downloaded(self):
        c = self._make()
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with patch("scroot.corrector.local.is_model_downloaded", return_value=False):
                with pytest.raises(RuntimeError, match="not downloaded"):
                    c._ensure_loaded()

    def test_ensure_loaded_creates_llama_instance(self):
        c = self._make()
        mock_llama = MagicMock()
        mock_mod = MagicMock()
        mock_mod.Llama = mock_llama
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with patch("scroot.corrector.local.is_model_downloaded", return_value=True):
                with patch("scroot.corrector.local.get_model_path") as mp:
                    mp.return_value = MagicMock(__str__=lambda _: "/fake/model.gguf")
                    c._ensure_loaded()
        assert c._llm is not None
        mock_llama.assert_called_once()

    def test_ensure_loaded_noop_when_already_loaded(self):
        c = self._make()
        sentinel = object()
        c._llm = sentinel
        c._ensure_loaded()
        assert c._llm is sentinel  # unchanged

    def test_draft_correction_uses_llm(self):
        c = self._make()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "  Fixed answer  "}}]
        }
        with patch.object(c, "_ensure_loaded"):
            c._llm = mock_llm
            result = c.draft_correction("What?", "Wrong.", "Context.")
        assert result == "Fixed answer"
        mock_llm.create_chat_completion.assert_called_once()

    def test_draft_correction_without_context(self):
        c = self._make()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Answer"}}]
        }
        with patch.object(c, "_ensure_loaded"):
            c._llm = mock_llm
            result = c.draft_correction("q", "r", None)
        assert result == "Answer"

    def test_build_prompt_includes_context(self):
        c = self._make()
        prompt = c._build_prompt("query", "response", "some context")
        assert "query" in prompt
        assert "response" in prompt
        assert "some context" in prompt

    def test_build_prompt_without_context(self):
        c = self._make()
        prompt = c._build_prompt("query", "response", None)
        assert "query" in prompt
        assert "response" in prompt
        assert "some context" not in prompt

    def test_unload_clears_llm(self):
        c = self._make()
        c._llm = MagicMock()
        c.unload()
        assert c._llm is None

    def test_unload_when_not_loaded(self):
        c = self._make()
        c.unload()  # should not raise
        assert c._llm is None

    def test_model_spec_returns_spec(self):
        c = self._make("phi4-mini")
        assert c.model_spec.id == "phi4-mini"
        assert c.model_spec.name == "Qwen2.5-3B-Instruct"

    def test_tok_per_sec_phi4(self):
        c = self._make("phi4-mini")
        assert c.tok_per_sec() == 16.0

    def test_tok_per_sec_smollm3(self):
        c = self._make("smollm3")
        assert c.tok_per_sec() == 22.0

    def test_tok_per_sec_unknown_model_returns_none(self):
        c = self._make("phi4-mini")
        c._config = MagicMock(model_id="unknown-model")
        assert c.tok_per_sec() is None

    def test_n_threads_defaults_to_cpu_count(self):
        c = self._make()
        mock_llama = MagicMock()
        mock_mod = MagicMock()
        mock_mod.Llama = mock_llama
        with patch.dict(sys.modules, {"llama_cpp": mock_mod}):
            with patch("scroot.corrector.local.is_model_downloaded", return_value=True):
                with patch("scroot.corrector.local.get_model_path") as mp:
                    mp.return_value = MagicMock(__str__=lambda _: "/m.gguf")
                    with patch("os.cpu_count", return_value=8):
                        c._ensure_loaded()
        _, kwargs = mock_llama.call_args
        assert kwargs.get("n_threads") == 8 or mock_llama.call_args[1].get("n_threads") == 8


# ─── APICorrector ─────────────────────────────────────────────────────────────

class TestAPICorrector:
    def _make(self, api_key="sk-test", model="gpt-4o-mini") -> APICorrector:
        return APICorrector(APIConfig(api_key=api_key, model=model))

    def test_is_available_true(self):
        assert self._make("sk-xyz").is_available is True

    def test_is_available_false_empty_key(self):
        assert self._make("").is_available is False

    def test_draft_correction_raises_without_httpx(self):
        c = self._make()
        with patch.dict(sys.modules, {"httpx": None}):
            with pytest.raises(RuntimeError, match="httpx"):
                c.draft_correction("q", "r", None)

    def test_draft_correction_calls_httpx(self):
        c = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "  Corrected  "}}]
        }
        mock_resp.raise_for_status = MagicMock()

        import httpx as _httpx
        with patch.object(_httpx, "post", return_value=mock_resp) as mock_post:
            result = c.draft_correction("What?", "Wrong.", "Context doc.")
        assert result == "Corrected"
        mock_post.assert_called_once()

    def test_draft_correction_anthropic_key(self):
        c = APICorrector(APIConfig(api_key="sk-ant-key123"))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        import httpx as _httpx
        with patch.object(_httpx, "post", return_value=mock_resp) as mock_post:
            result = c.draft_correction("q", "r", None)
        assert result == "Answer"
        call_kwargs = mock_post.call_args
        # Anthropic should use x-api-key header
        headers = call_kwargs[1]["headers"] if "headers" in call_kwargs[1] else call_kwargs[0][1]
        assert "x-api-key" in headers or any("anthropic" in str(v) for v in call_kwargs[1].values())

    def test_build_prompt_with_context(self):
        c = self._make()
        result = c._build_prompt("q", "r", "ctx")
        assert "ctx" in result

    def test_build_prompt_without_context(self):
        c = self._make()
        result = c._build_prompt("q", "r", None)
        assert "ctx" not in result

    def test_draft_correction_with_base_url_override(self, monkeypatch):
        # M-2: a custom (non-provider) base_url requires the explicit opt-in.
        monkeypatch.setenv("SCROOT_ALLOW_ANY_BASE_URL", "1")
        c = APICorrector(APIConfig(api_key="sk-test", base_url="https://custom.example.com/v1"))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        import httpx as _httpx
        with patch.object(_httpx, "post", return_value=mock_resp) as mock_post:
            c.draft_correction("q", "r", None)
        url = mock_post.call_args[0][0]
        assert "custom.example.com" in url

    def test_draft_correction_untrusted_base_url_blocked(self):
        # M-2: without the opt-in, an untrusted endpoint is refused before any
        # request is made (the key is never sent).
        c = APICorrector(APIConfig(api_key="sk-test", base_url="https://evil.example.com/v1"))
        import httpx as _httpx
        with patch.object(_httpx, "post") as mock_post:
            with pytest.raises(ValueError):
                c.draft_correction("q", "r", None)
        mock_post.assert_not_called()
