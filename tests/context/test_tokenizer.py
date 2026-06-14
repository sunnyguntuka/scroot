"""Token counting tests - tiktoken path and charcount fallback."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import scroot.context.tokenizer as tok


class TestFallback:
    def test_empty_string_is_zero(self):
        assert tok.count_tokens("") == 0

    def test_charcount_fallback(self, monkeypatch):
        monkeypatch.setattr(tok, "_tiktoken_checked", True)
        monkeypatch.setattr(tok, "_encoder", None)
        assert tok.count_tokens("abcd") == 1       # 4 chars -> 1 token
        assert tok.count_tokens("abcde") == 2      # ceil(5/4)
        assert tok.count_tokens("x") == 1          # never zero for non-empty


class TestTiktokenPath:
    def test_uses_tiktoken_when_available(self, monkeypatch):
        fake_encoder = MagicMock()
        fake_encoder.encode.return_value = [1, 2, 3, 4, 5]
        fake_tiktoken = MagicMock()
        fake_tiktoken.get_encoding.return_value = fake_encoder

        monkeypatch.setattr(tok, "_tiktoken_checked", False)
        monkeypatch.setattr(tok, "_encoder", None)
        with patch.dict(sys.modules, {"tiktoken": fake_tiktoken}):
            assert tok.count_tokens("hello world") == 5
        fake_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    def test_broken_tiktoken_falls_back(self, monkeypatch):
        fake_tiktoken = MagicMock()
        fake_tiktoken.get_encoding.side_effect = RuntimeError("no data files")

        monkeypatch.setattr(tok, "_tiktoken_checked", False)
        monkeypatch.setattr(tok, "_encoder", None)
        with patch.dict(sys.modules, {"tiktoken": fake_tiktoken}):
            assert tok.count_tokens("abcd") == 1
