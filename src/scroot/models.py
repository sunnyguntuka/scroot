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


# ---------------------------------------------------------------------------
# MiniCheck backbone
# ---------------------------------------------------------------------------

class MiniCheckRobertaBackbone:
    """MiniCheck-RoBERTa-Large binary support classifier.

    Drop-in groundedness backend for score_groundedness(). score_pairs()
    returns P(supported) directly — treated as the entailment probability.
    Similarity fallback is skipped (not applicable to binary classifiers).
    """

    HF_NAME = "lytang/MiniCheck-RoBERTa-Large"
    _BATCH_SIZE = 16

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self._torch = torch
        self._device = device
        self._tok = AutoTokenizer.from_pretrained(self.HF_NAME)
        self._model = (AutoModelForSequenceClassification
                       .from_pretrained(self.HF_NAME)
                       .to(device)
                       .eval())

    def score_pairs(self, pairs: "list[tuple[str, str]]") -> "list[float]":
        """Return P(claim supported by premise) for each (premise, claim) pair."""
        out: list[float] = []
        for i in range(0, len(pairs), self._BATCH_SIZE):
            batch = pairs[i:i + self._BATCH_SIZE]
            docs = [p[0] for p in batch]
            claims = [p[1] for p in batch]
            enc = self._tok(docs, claims, truncation=True, max_length=512,
                            padding=True, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                logits = self._model(**enc).logits
                probs = self._torch.softmax(logits, dim=-1)[:, 1]
            out.extend(probs.cpu().tolist())
        return out


_BACKBONE_REGISTRY: dict[str, type] = {
    "minicheck-roberta-large": MiniCheckRobertaBackbone,
}

# Canonical names that map to the standard deberta NLI path (backbone=None)
_DEBERTA_ALIASES: frozenset[str] = frozenset({
    "deberta-base",
    "cross-encoder/nli-deberta-v3-base",
    "deberta-small",
    "cross-encoder/nli-deberta-v3-small",
    "deberta-large",
    "cross-encoder/nli-deberta-v3-large",
})


def get_groundedness_backbone(name: str, device: str = "cpu"):
    """Return a backbone scorer instance, or None for the deberta NLI path.

    ``None`` means score_groundedness() uses its standard
    ``get_nli_model`` / softmax / similarity-fallback path.
    A returned object must expose ``score_pairs(pairs) -> list[float]``
    where each float is P(claim supported by premise) in [0, 1].

    Args:
        name: Backbone identifier. ``"deberta-base"`` (and its aliases)
            return ``None``. ``"minicheck-roberta-large"`` returns a
            cached :class:`MiniCheckRobertaBackbone`.
        device: Inference device passed to the backbone constructor.

    Raises:
        ValueError: If ``name`` is not a known backbone identifier.
    """
    if name in _DEBERTA_ALIASES:
        return None

    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown groundedness backbone {name!r}. "
            f"Supported: {sorted(_DEBERTA_ALIASES | set(_BACKBONE_REGISTRY))}."
        )

    key = f"backbone:{name}:{device}"
    with _cache_lock:
        if key in _model_cache:
            _model_cache.move_to_end(key)
            return _model_cache[key]
        if len(_model_cache) >= MAX_CACHE_SIZE:
            _model_cache.popitem(last=False)
        _model_cache[key] = _BACKBONE_REGISTRY[name](device=device)
        return _model_cache[key]
