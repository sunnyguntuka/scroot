"""Scroot exception and warning types.

Warning policy: warn and degrade on content errors, fail hard only on
programming errors (such as mutating a sealed ContextBuilder).
"""

from __future__ import annotations


class ContextEmptyWarning(UserWarning):
    """build() called with no content added. Groundedness will be None."""


class ContextTooLargeWarning(UserWarning):
    """Context exceeded max_tokens. Lower-priority content was truncated."""


class ContextAssemblyWarning(UserWarning):
    """Content was deduplicated, skipped, or modified during assembly."""


class ContextSealedError(RuntimeError):
    """Content added to a ContextBuilder after build() was called."""


class ContextTypeError(TypeError):
    """Unrecognised input type passed to add_retrieved() or add_reranked()."""


class SecurityWarning(UserWarning):
    """A security-relevant default was disabled (e.g. pii_scrub=False in production)."""


class ModelDownloadError(RuntimeError):
    """Failed to download or load a scoring model (network, disk, or cache issue)."""


class NoContextWarning(UserWarning):
    """auditor.score() was called without context.

    Groundedness cannot be computed, so it is None and excluded from IQS - the
    score is the weighted harmonic mean of the remaining 4 metrics (weights
    redistributed). Provide context= or use ContextBuilder to score groundedness.
    """


class GroundednessUnavailableWarning(UserWarning):
    """A groundedness floor was requested via passes_gate()/gate_reason() but
    groundedness is None (no context provided).

    The floor could not be evaluated; the gate fails open (does not reject the
    response on this floor alone). Provide context to enforce the requirement.
    """


class GroundednessComputationError(UserWarning):
    """Context was provided but groundedness scoring raised an unexpected error.

    Groundedness is set to None and excluded from IQS; the other four metrics
    still score. Emitted as a warning so a model/runtime error degrades
    gracefully instead of failing the whole call.
    """
