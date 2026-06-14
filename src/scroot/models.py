"""Lazy model loading with singleton caching.

sentence-transformers (and PyTorch) are imported lazily inside the loader
functions, so ``import scroot`` is fast (<100ms) and does not fail when
sentence-transformers is not installed. A clear ImportError is raised only
when a model is first loaded (Audit 6).

Only model names on the trusted allowlist are loaded -callers must call
trust_model() to authorise a custom name (C-2: pickle RCE mitigation).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from .exceptions import ModelDownloadError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Trusted model allowlist
# ---------------------------------------------------------------------------

DEFAULT_ALLOWED_MODELS: frozenset[str] = frozenset({
    # NLI cross-encoders (accuracy in ascending order)
    "cross-encoder/nli-deberta-v3-small",   # fastest, lowest accuracy
    "cross-encoder/nli-deberta-v3-base",    # default: best speed/accuracy tradeoff
    "cross-encoder/nli-deberta-v3-large",   # +4% accuracy, ~2x slower
    # Embedding models
    "all-MiniLM-L6-v2",                    # default: fast, 90MB
    "all-MiniLM-L12-v2",                   # slightly better, same size
    "all-mpnet-base-v2",                    # highest quality, 420MB
})

_allowed_models: set[str] = set(DEFAULT_ALLOWED_MODELS)
_allowlist_lock = threading.Lock()

MAX_CACHE_SIZE = 10


def validate_model_name(name: str) -> None:
    """Raise ValueError if name is not on the trusted model allowlist.

    Args:
        name: HuggingFace model identifier string.

    Raises:
        ValueError: If the model is not in the allowlist.
    """
    with _allowlist_lock:
        allowed = set(_allowed_models)
    if name not in allowed:
        raise ValueError(
            f"Model {name!r} is not on the trusted allowlist. "
            f"Call scroot.models.trust_model({name!r}) to explicitly "
            f"authorize it before use."
        )


def trust_model(name: str) -> None:
    """Add a model name to the runtime trusted allowlist.

    Call this once at application startup to authorize a custom or
    fine-tuned model that is not in DEFAULT_ALLOWED_MODELS.

    Args:
        name: HuggingFace model identifier to trust.
    """
    with _allowlist_lock:
        _allowed_models.add(name)


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_model_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def _download_help(model_name: str, loader_snippet: str, error: Exception) -> str:
    return (
        f"Failed to download or load model {model_name!r}.\n"
        f"Check your internet connection and try again.\n"
        f"To use scroot offline, pre-download the model:\n"
        f"  python -c \"{loader_snippet}\"\n"
        f"Original error: {error}"
    )


def get_embedding_model(
    model_name_or_instance: "str | SentenceTransformer",
    device: str = "cpu",
) -> "SentenceTransformer":
    """Load or retrieve a cached embedding model.

    sentence-transformers is imported lazily on first call so that
    ``import scroot`` remains fast and does not require the package to be
    installed until a model is actually needed.

    Accepts either a model-name string (validated against the allowlist)
    or a pre-instantiated SentenceTransformer (bypasses validation -the
    caller already loaded the model and is responsible for its safety).

    Args:
        model_name_or_instance: HuggingFace model name string, or an
            already-loaded SentenceTransformer instance.
        device: "cpu" or "cuda".

    Returns:
        SentenceTransformer instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
        ValueError: If model_name_or_instance is a string not on the allowlist.
        ModelDownloadError: If the model weights fail to download or load
            (network error, disk full, interrupted download, etc.).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding models. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    if isinstance(model_name_or_instance, SentenceTransformer):
        return model_name_or_instance

    model_name = model_name_or_instance
    validate_model_name(model_name)
    key = f"emb:{model_name}:{device}"
    with _cache_lock:
        if key in _model_cache:
            _model_cache.move_to_end(key)
            return _model_cache[key]
        if len(_model_cache) >= MAX_CACHE_SIZE:
            _model_cache.popitem(last=False)
        try:
            _model_cache[key] = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise ModelDownloadError(_download_help(
                model_name,
                f"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer({model_name!r})",
                exc,
            )) from exc
        return _model_cache[key]


def get_nli_model(
    model_name_or_instance: "str | CrossEncoder",
    device: str = "cpu",
) -> "CrossEncoder":
    """Load or retrieve a cached NLI cross-encoder model.

    sentence-transformers is imported lazily on first call (Audit 6).

    Accepts either a model-name string (validated against the allowlist)
    or a pre-instantiated CrossEncoder (bypasses validation).

    Args:
        model_name_or_instance: HuggingFace model name string, or an
            already-loaded CrossEncoder instance.
        device: "cpu" or "cuda".

    Returns:
        CrossEncoder instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
        ValueError: If model_name_or_instance is a string not on the allowlist.
        ModelDownloadError: If the model weights fail to download or load
            (network error, disk full, interrupted download, etc.).
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for NLI models. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    if isinstance(model_name_or_instance, CrossEncoder):
        return model_name_or_instance

    model_name = model_name_or_instance
    validate_model_name(model_name)
    key = f"nli:{model_name}:{device}"
    with _cache_lock:
        if key in _model_cache:
            _model_cache.move_to_end(key)
            return _model_cache[key]
        if len(_model_cache) >= MAX_CACHE_SIZE:
            _model_cache.popitem(last=False)
        try:
            _model_cache[key] = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise ModelDownloadError(_download_help(
                model_name,
                f"from sentence_transformers import CrossEncoder; "
                f"CrossEncoder({model_name!r})",
                exc,
            )) from exc
        return _model_cache[key]


def clear_cache() -> None:
    """Clear all cached models."""
    with _cache_lock:
        _model_cache.clear()
