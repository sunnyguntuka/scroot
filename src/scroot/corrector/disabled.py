"""NullCorrector - no LLM call, returns None."""
from __future__ import annotations

from scroot.corrector.base import BaseCorrector


class NullCorrector(BaseCorrector):
    @property
    def is_available(self) -> bool:
        return False

    def draft_correction(self, query: str, response: str, context: str | None) -> None:
        return None
