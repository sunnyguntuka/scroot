"""ContextEntry and ContextPayload dataclasses.

ContextPayload is what auditor.score() receives. It stores the assembled
(scrubbed) text and the audit trail - never the raw pre-scrub additions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ContextEntry:
    """A single piece of context added to a ContextBuilder.

    Attributes:
        source: Source label - 'retrieval', 'reranker', 'system_prompt',
            'tool_output', 'query', or a custom label.
        content: Scrubbed content (PII already replaced if pii_scrub=True).
        added_at: UTC timestamp of the addition.
        metadata: Caller-supplied metadata. Audit-trail only, not scored.
        source_weight: 0.0-1.0; higher = more authoritative for groundedness.
        token_count: Token count of content.
        was_scrubbed: True if PII was detected and replaced in this entry.
        scrub_summary: Entity type counts only - no original values.
    """
    source: str
    content: str
    added_at: datetime
    metadata: dict = field(default_factory=dict)
    source_weight: float = 0.6
    token_count: int = 0
    was_scrubbed: bool = False
    scrub_summary: dict = field(default_factory=dict)


@dataclass
class ContextPayload:
    """Assembled context returned by ContextBuilder.build().

    Pass this to ``auditor.score(context=...)``. The payload is consumed
    during scoring - the assembled text feeds the NLI scorer locally and
    is then discarded. Only ``session_id`` and ``checksum`` flow into
    downstream records for audit-trail purposes.

    Attributes:
        session_id: Trace identifier from the originating ContextBuilder.
        sources: The kept ContextEntry items, highest-weight first.
        assembled_text: Final concatenated grounding string (scrubbed).
        total_tokens: Token count of the kept entries.
        was_truncated: True if the max_tokens budget dropped entries.
        pii_scrubbed: True if any kept entry had PII replaced.
        scrub_summary: Aggregated entity-type counts (no original text).
        built_at: UTC timestamp when build() was called.
        checksum: ``sha256:<hex>`` of assembled_text for integrity checks.
    """
    session_id: str
    sources: list[ContextEntry]
    assembled_text: str
    total_tokens: int
    was_truncated: bool
    pii_scrubbed: bool
    scrub_summary: dict
    built_at: datetime
    checksum: str
