"""Guardrail injector: builds correction context for LLM prompts.

Reads the feedback store and produces a text block that can be
injected into a system prompt to prevent repeated mistakes.

Three strategies:
1. Recent corrections -raw recent corrections, most flexible
2. Rule extraction -collapsed patterns, most token-efficient
3. Query-relevant -embedding search, most targeted

All record fields are sanitized with sanitize_for_prompt() before
interpolation to prevent stored prompt-injection attacks (C-1).
PII patterns (SSN, email, phone, credit card) are scrubbed before
injection to prevent cross-user data leakage (H-2).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from .sanitizer import sanitize_for_prompt

if TYPE_CHECKING:
    from .store import FeedbackStore

# --- PII scrubbing patterns ---------------------------------------------------

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE]"),
    (
        re.compile(
            r"\b(?:"
            r"4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4}"  # Visa 13-16 digit
            r"|5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"  # MC 16 digit
            r"|3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}"  # Amex 15 digit
            r"|3(?:0[0-5]|[68]\d)\d{2}[\s\-]?\d{6}[\s\-]?\d{4}"  # Diners 14 digit
            r"|6(?:011|5\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"  # Discover 16 digit
            r")\b"
        ),
        "[CARD]",
    ),
]


def default_pii_scrubber(text: str) -> str:
    """Mask SSNs, emails, phone numbers, and credit-card numbers."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# --- Injector -----------------------------------------------------------------


class GuardrailInjector:
    """Builds guardrail context from past corrections.

    Args:
        store: FeedbackStore instance.
        pii_scrubber: Optional callable(str) -> str applied to each field
            before prompt interpolation. Defaults to default_pii_scrubber
            which masks SSNs, emails, phone numbers, and credit cards.
            Pass ``None`` to disable PII scrubbing.
    """

    def __init__(
        self,
        store: "FeedbackStore",
        pii_scrubber: "Callable[[str], str] | None" = default_pii_scrubber,
    ):
        self.store = store
        self._pii_scrubber = pii_scrubber

    def _clean(self, text: str, max_length: int) -> str:
        """Apply PII scrubbing then prompt-injection sanitization."""
        if self._pii_scrubber is not None:
            text = self._pii_scrubber(text)
        return sanitize_for_prompt(text, max_length=max_length)

    def build_context(
        self,
        query: str | None = None,
        strategy: str = "relevant",
        max_corrections: int = 5,
        max_tokens: int = 500,
    ) -> str:
        """Build a guardrail text block for system prompt injection.

        Args:
            query: Current user query (needed for "relevant" strategy).
            strategy: "recent", "relevant", or "rules".
            max_corrections: Max corrections to include.
            max_tokens: Approximate token budget (1 token ≈ 4 chars).

        Returns:
            Formatted string ready for system prompt injection, with all
            record fields sanitized against prompt-injection and PII leakage.
        """
        if strategy == "recent":
            records = self.store.get_recent(max_corrections)
        elif strategy == "relevant":
            if query is None:
                raise ValueError("query required for 'relevant' strategy")
            records = self.store.search(query, top_k=max_corrections)
        elif strategy == "rules":
            return self._build_rules(max_tokens)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        if not records:
            return ""

        lines = ["[KNOWN CORRECTIONS -do not repeat these mistakes]"]
        char_budget = max_tokens * 4
        included_ids = []

        for r in records:
            entry = (
                f"- Query: {self._clean(r.query, 200)}\n"
                f"  Wrong: {self._clean(r.response, 200)}\n"
                f"  Correct: {self._clean(r.correction, 200)}\n"
                f"  Reason: {self._clean(r.reason, 150)}"
            )
            if len("\n".join(lines)) + len(entry) > char_budget:
                break
            lines.append(entry)
            included_ids.append(r.id)

        self.store.increment_guardrail_count(included_ids)
        return "\n".join(lines)

    def _build_rules(self, max_tokens: int) -> str:
        """Extract collapsed rules from correction patterns."""
        records = self.store.get_all()
        if not records:
            return ""

        seen_reasons: set[str] = set()
        rules = []
        for r in records:
            reason_key = r.reason.lower().strip()[:80]
            if reason_key not in seen_reasons:
                seen_reasons.add(reason_key)
                rules.append((f"- {self._clean(r.reason, 200)}", r.id))

        header = "[GUARDRAILS -rules extracted from past corrections]"
        char_budget = max_tokens * 4
        lines = [header]
        included_ids = []
        for rule, record_id in rules:
            if len("\n".join(lines)) + len(rule) > char_budget:
                break
            lines.append(rule)
            included_ids.append(record_id)

        self.store.increment_guardrail_count(included_ids)
        return "\n".join(lines)
