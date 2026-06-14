"""Tests for graceful model-download failure (ModelDownloadError)."""

import pytest

from scroot.exceptions import ModelDownloadError
from scroot.models import clear_cache, get_embedding_model, get_nli_model


@pytest.fixture(autouse=True)
def _clear_model_cache():
    clear_cache()
    yield
    clear_cache()


def test_get_embedding_model_wraps_construction_failure(monkeypatch):
    sentence_transformers = pytest.importorskip("sentence_transformers")

    class Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("network is unreachable")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Boom)

    with pytest.raises(ModelDownloadError) as excinfo:
        get_embedding_model("all-MiniLM-L6-v2")

    message = str(excinfo.value)
    assert "all-MiniLM-L6-v2" in message
    assert "SentenceTransformer" in message
    assert "network is unreachable" in message
    assert isinstance(excinfo.value.__cause__, OSError)


def test_get_nli_model_wraps_construction_failure(monkeypatch):
    sentence_transformers = pytest.importorskip("sentence_transformers")

    class Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("connection timed out")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Boom)

    with pytest.raises(ModelDownloadError) as excinfo:
        get_nli_model("cross-encoder/nli-deberta-v3-base")

    message = str(excinfo.value)
    assert "cross-encoder/nli-deberta-v3-base" in message
    assert "CrossEncoder" in message
    assert "connection timed out" in message
    assert isinstance(excinfo.value.__cause__, OSError)


def test_get_embedding_model_import_error_not_wrapped(monkeypatch):
    """Missing sentence-transformers package stays a plain ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="sentence-transformers is required"):
        get_embedding_model("all-MiniLM-L6-v2")
