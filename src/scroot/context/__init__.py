"""Context assembly for groundedness scoring.

Public exports: ContextBuilder, ContextPayload, ContextEntry.
"""

from .builder import ContextBuilder
from .payload import ContextEntry, ContextPayload

__all__ = ["ContextBuilder", "ContextPayload", "ContextEntry"]
