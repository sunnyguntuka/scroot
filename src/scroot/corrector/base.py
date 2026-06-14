"""Base corrector ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCorrector(ABC):
    @abstractmethod
    def draft_correction(
        self,
        query: str,
        response: str,
        context: str | None,
    ) -> str | None:
        """Return a correction draft, or None if disabled."""

    @property
    def is_available(self) -> bool:
        """True if this corrector can generate drafts right now."""
        return True
