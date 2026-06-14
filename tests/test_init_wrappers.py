"""Tests for top-level convenience wrappers (score, verify, setup_nltk)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import scroot


class StubResult:
    def __init__(self, iqs):
        self.iqs = iqs


class StubAuditor:
    """Stands in for Auditor - records constructor kwargs and score args."""

    last_kwargs = None
    last_score_args = None
    next_iqs = 0.9

    def __init__(self, **kwargs):
        StubAuditor.last_kwargs = kwargs

    def score(self, query, response, context=None):
        StubAuditor.last_score_args = (query, response, context)
        return StubResult(StubAuditor.next_iqs)


class TestScoreWrapper:
    def test_score_delegates_to_auditor(self, monkeypatch):
        monkeypatch.setattr(scroot, "Auditor", StubAuditor)
        result = scroot.score("q", "r", context=["c"], device="cpu")
        assert result.iqs == 0.9
        assert StubAuditor.last_kwargs == {"device": "cpu"}
        assert StubAuditor.last_score_args == ("q", "r", ["c"])


class TestVerifyWrapper:
    def test_verify_passes_at_threshold(self, monkeypatch):
        monkeypatch.setattr(scroot, "Auditor", StubAuditor)
        StubAuditor.next_iqs = 0.7
        assert scroot.verify("q", "r", threshold=0.7) is True

    def test_verify_fails_below_threshold(self, monkeypatch):
        monkeypatch.setattr(scroot, "Auditor", StubAuditor)
        StubAuditor.next_iqs = 0.69
        assert scroot.verify("q", "r", threshold=0.7) is False


class TestSetupNltk:
    def test_downloads_punkt_tab(self):
        fake_nltk = MagicMock()
        with patch.dict(sys.modules, {"nltk": fake_nltk}):
            scroot.setup_nltk()
        fake_nltk.download.assert_called_once_with("punkt_tab", quiet=False)
