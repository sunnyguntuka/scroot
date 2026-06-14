"""ContextBuilder - request-scoped context accumulator.

Carries grounding documents through a multi-step RAG or agentic pipeline
and delivers them to ``auditor.score()`` intact, without restructuring
the client's code.

SOC II posture: content is held in memory only, PII-scrubbed by default,
and never written to disk. Audit events are content-free. Only
floating-point scores cross tier boundaries.
"""

from __future__ import annotations

import hashlib
import os
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

from .. import audit
from ..exceptions import (
    ContextAssemblyWarning,
    ContextEmptyWarning,
    ContextSealedError,
    ContextTooLargeWarning,
    SecurityWarning,
)
from .adapters import extract_text
from .dedup import deduplicate
from .payload import ContextEntry, ContextPayload
from .pii import scrub
from .tokenizer import count_tokens

_SOURCE_WEIGHTS: dict[str, float] = {
    "reranker":      1.0,
    "retrieval":     0.85,
    "tool_output":   0.70,
    "system_prompt": 0.50,
    "query":         0.30,
    "custom":        0.60,
}

_MAX_CHUNK_CHARS = 50_000
_MAX_CHUNKS_PER_CALL = 500
_MAX_SESSION_ID_LEN = 128
_MAX_METADATA_KEYS = 20
_MAX_METADATA_VALUE_CHARS = 1_000


class ContextBuilder:
    """Accumulates grounding context across a multi-step LLM pipeline.

    Create one per request, add grounding material as it becomes
    available at each pipeline step, and pass ``ctx.build()`` to
    ``auditor.score(context=...)`` at the end. The client's LLM call is
    never touched.

    Example:
        >>> import scroot
        >>> ctx = scroot.ContextBuilder()
        >>> ctx.add_query(user_query)
        >>> ctx.add_retrieved(retriever.search(user_query))
        >>> result = auditor.score(query, response, context=ctx.build())

    Args:
        session_id: Ties this context to a trace; auto-generated UUID4
            if omitted. Max 128 chars.
        max_tokens: Hard ceiling on assembled context size. build()
            truncates lowest-priority sources and emits
            ContextTooLargeWarning when exceeded. Default 4096.
        pii_scrub: Run PII detection before storing each addition
            (default True). Detected entities are replaced with typed
            placeholders ([EMAIL], [PHONE], [SECRET], ...). The audit
            trail records counts only, never the original values.
            Disabling in production (SCROOT_ENV=production) emits a
            SecurityWarning.
        dedup: Deduplicate overlapping chunk content on build() using
            cosine similarity at the 0.92 threshold (default True).
        encryption_key: Fernet key for encrypting context at rest if a
            session store is configured. With the default None, content
            is held in memory only - nothing is written to disk, so no
            encryption is needed.

    SOC II: content is held in memory only, PII-scrubbed by default,
    never written to disk unless encryption_key is provided.
    """

    def __init__(
        self,
        session_id: str | None = None,
        max_tokens: int = 4096,
        pii_scrub: bool = True,
        dedup: bool = True,
        encryption_key: bytes | None = None,
    ) -> None:
        if session_id is not None and len(session_id) > _MAX_SESSION_ID_LEN:
            raise ValueError(
                f"session_id exceeds {_MAX_SESSION_ID_LEN} chars."
            )
        if not pii_scrub and os.environ.get("SCROOT_ENV") == "production":
            warnings.warn(
                "pii_scrub=False with SCROOT_ENV=production. "
                "PII in context content will not be redacted.",
                SecurityWarning,
                stacklevel=2,
            )
        if encryption_key is not None:
            try:
                from cryptography.fernet import Fernet
                Fernet(encryption_key)  # validates the key format
            except ImportError as exc:
                raise ImportError(
                    "encryption_key requires the cryptography package: "
                    "pip install 'scroot[security]'"
                ) from exc

        self._session_id = session_id or f"cb-{uuid.uuid4()}"
        self._max_tokens = max_tokens
        self._pii_scrub = pii_scrub
        self._dedup = dedup
        self._encryption_key = encryption_key
        self._entries: list[ContextEntry] = []
        self._sealed = False
        self._built_at: datetime | None = None

    @property
    def session_id(self) -> str:
        """The trace identifier for this builder."""
        return self._session_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _guard_sealed(self) -> None:
        if self._sealed:
            raise ContextSealedError(
                "ContextBuilder has been sealed by build(). "
                "Create a new ContextBuilder for each request."
            )

    @staticmethod
    def _validate_metadata(metadata: dict) -> None:
        if len(metadata) > _MAX_METADATA_KEYS:
            raise ValueError(
                f"metadata exceeds {_MAX_METADATA_KEYS} keys."
            )
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) > _MAX_METADATA_VALUE_CHARS:
                raise ValueError(
                    f"metadata value for {key!r} exceeds "
                    f"{_MAX_METADATA_VALUE_CHARS} chars."
                )

    def _process_text(
        self, text: str, source: str, metadata: dict
    ) -> ContextEntry | None:
        if not text or not text.strip():
            return None

        if len(text) > _MAX_CHUNK_CHARS:
            text = text[:_MAX_CHUNK_CHARS] + " [TRUNCATED]"
            warnings.warn(
                f"Chunk from source '{source}' exceeded "
                f"{_MAX_CHUNK_CHARS:,} chars and was truncated.",
                ContextAssemblyWarning,
                stacklevel=4,
            )

        scrub_summary: dict = {}
        was_scrubbed = False
        if self._pii_scrub:
            try:
                result = scrub(text)
                text = result.scrubbed_text
                scrub_summary = result.summary
                was_scrubbed = result.was_scrubbed
            except Exception:
                warnings.warn(
                    "PII scrubber failed; content passed through unscrubbed.",
                    ContextAssemblyWarning,
                    stacklevel=4,
                )

        return ContextEntry(
            source=source,
            content=text,
            added_at=datetime.now(timezone.utc),
            metadata=metadata,
            source_weight=_SOURCE_WEIGHTS.get(source, 0.60),
            token_count=count_tokens(text),
            was_scrubbed=was_scrubbed,
            scrub_summary=scrub_summary,
        )

    def _add_chunks(
        self, chunks: Any, source: str, metadata: dict
    ) -> "ContextBuilder":
        self._guard_sealed()
        self._validate_metadata(metadata)

        if isinstance(chunks, str):
            chunks = [chunks]
        elif isinstance(chunks, dict) or not hasattr(chunks, '__iter__'):
            chunks = [chunks]

        chunks = list(chunks)
        if len(chunks) > _MAX_CHUNKS_PER_CALL:
            warnings.warn(
                f"Received {len(chunks)} chunks for source '{source}'; "
                f"only the first {_MAX_CHUNKS_PER_CALL} will be used.",
                ContextAssemblyWarning,
                stacklevel=3,
            )
            chunks = chunks[:_MAX_CHUNKS_PER_CALL]

        added: list[ContextEntry] = []
        for chunk in chunks:
            text = extract_text(chunk)
            if text is None:
                warnings.warn(
                    f"Could not extract text from chunk of type "
                    f"'{type(chunk).__name__}' in source '{source}'. Skipped.",
                    ContextAssemblyWarning,
                    stacklevel=3,
                )
                continue
            entry = self._process_text(text, source, metadata)
            if entry:
                self._entries.append(entry)
                added.append(entry)

        if added:
            scrub_totals: dict[str, int] = {}
            for entry in added:
                for k, v in entry.scrub_summary.items():
                    if v:
                        scrub_totals[k] = scrub_totals.get(k, 0) + v
            audit.emit(
                "context_entry_added",
                session_id=self._session_id,
                source=source,
                token_count=sum(e.token_count for e in added),
                chunk_count=len(added),
                pii_scrubbed=any(e.was_scrubbed for e in added),
                scrub_summary=scrub_totals,
            )
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_query(
        self, text: str, *, metadata: dict | None = None
    ) -> "ContextBuilder":
        """Record the user's query. Call first, before retrieval.

        Calling more than once appends to query history with timestamps —
        useful for multi-turn conversations where the query evolves.

        Args:
            text: The user's query. Plain string only.
            metadata: Optional dict, audit-trail only. Max 20 keys.

        Returns:
            self, for method chaining.

        Raises:
            ContextSealedError: If called after build().
        """
        return self._add_chunks(text, "query", metadata or {})

    def add_retrieved(
        self,
        chunks: Any,
        *,
        source: str = "retrieval",
        metadata: dict | None = None,
    ) -> "ContextBuilder":
        """Record retrieved documents for groundedness scoring.

        Call this immediately after your retrieval step, before any
        reranking or LLM call. This is the most important method - it's
        what gives groundedness its signal.

        Args:
            chunks: Retrieved documents. See supported types below.
            source: Label for this retrieval source. Used in audit logs
                and dashboard provenance display. Defaults to
                "retrieval". Use descriptive names for multi-source
                pipelines: "pinecone", "web_search", "internal_db".
            metadata: Optional dict for additional context. Stored in
                audit log only - not used in scoring. Max 20 keys.

        Returns:
            self, for method chaining.

        Supported chunk types:
            - str: treated as a single chunk
            - list[str]: each string is a chunk
            - list[Document]: LangChain Documents (page_content extracted)
            - list[dict]: dicts with 'text', 'content', or 'page_content' key
            - QueryResult: ChromaDB result objects
            - list[ScoredVector]: Pinecone results (metadata['text'] extracted)

        Warns:
            ContextAssemblyWarning: If a chunk type is unrecognised
                (skipped, not raised - pipeline continues), or if more
                than 500 chunks are passed (excess dropped).

        Raises:
            ContextSealedError: If called after build().

        Example:
            >>> ctx = ContextBuilder()
            >>> ctx.add_query("What is the refund policy?")
            >>> docs = retriever.get_relevant_documents(query)
            >>> ctx.add_retrieved(docs)
            >>> result = auditor.score(query, response, context=ctx.build())
        """
        return self._add_chunks(chunks, source, metadata or {})

    def add_reranked(
        self,
        chunks: Any,
        *,
        source: str = "reranker",
        metadata: dict | None = None,
    ) -> "ContextBuilder":
        """Record post-reranking documents. Higher weight than raw retrieved.

        Reranked chunks carry higher weight in groundedness scoring than
        raw retrieved chunks, because they represent what the LLM
        actually used. Same accepted types as :meth:`add_retrieved`.

        Args:
            chunks: Post-reranking documents.
            source: Source label, defaults to "reranker".
            metadata: Optional dict, audit-trail only. Max 20 keys.

        Returns:
            self, for method chaining.

        Raises:
            ContextSealedError: If called after build().
        """
        return self._add_chunks(chunks, source, metadata or {})

    def add_system_prompt(
        self, text: str, *, metadata: dict | None = None
    ) -> "ContextBuilder":
        """Record the system prompt used in the LLM call.

        Included in groundedness scoring with lower weight than retrieved
        chunks - it's instructions, not facts.

        Args:
            text: The system prompt text.
            metadata: Optional dict, audit-trail only. Max 20 keys.

        Returns:
            self, for method chaining.

        Raises:
            ContextSealedError: If called after build().
        """
        return self._add_chunks(text, "system_prompt", metadata or {})

    def add_tool_output(
        self,
        output: str | list[str],
        *,
        tool_name: str,
        metadata: dict | None = None,
    ) -> "ContextBuilder":
        """Record a tool call output (DB query result, API response, etc.).

        Args:
            output: Tool output text, or a list of output strings.
            tool_name: Name of the tool that produced the output.
                Recorded in entry metadata and audit logs.
            metadata: Optional dict, audit-trail only. Max 20 keys.

        Returns:
            self, for method chaining.

        Raises:
            ContextSealedError: If called after build().
        """
        meta = {**(metadata or {}), "tool_name": tool_name}
        return self._add_chunks(output, "tool_output", meta)

    def snapshot(self) -> dict:
        """Return current state without building. For debugging/logging.

        Returns:
            Dict with session_id, sealed flag, source labels, entry and
            token counts, and whether PII scrubbing is enabled. Contains
            no content text.
        """
        return {
            "session_id":        self._session_id,
            "sealed":            self._sealed,
            "sources":           [e.source for e in self._entries],
            "total_entries":     len(self._entries),
            "total_tokens":      sum(e.token_count for e in self._entries),
            "pii_scrub_enabled": self._pii_scrub,
        }

    def reset(self) -> "ContextBuilder":
        """Clear all entries and unseal. Prefer a new instance per request.

        Returns:
            self, for method chaining.
        """
        self._entries.clear()
        self._sealed = False
        self._built_at = None
        return self

    def build(self) -> ContextPayload | None:
        """Assemble all context into a ContextPayload for auditor.score().

        Seals the builder - no further additions after this call.

        Assembly steps: sort entries by source weight
        (reranked > retrieved > tool_output > system_prompt > query),
        deduplicate near-identical chunks if dedup=True, then truncate
        to max_tokens keeping the highest-priority sources.

        Returns:
            ContextPayload, or None if nothing was added (groundedness
            will score as None with a warning - not a crash).

        Warns:
            ContextEmptyWarning: If no content was added.
            ContextTooLargeWarning: If max_tokens forced truncation.
        """
        self._sealed = True
        self._built_at = datetime.now(timezone.utc)

        if not self._entries:
            warnings.warn(
                "ContextBuilder.build() called with no content. "
                "Groundedness will be None. "
                "Call add_retrieved() before build() for full scoring.",
                ContextEmptyWarning,
                stacklevel=2,
            )
            return None

        sorted_entries = sorted(
            self._entries, key=lambda e: e.source_weight, reverse=True
        )

        if self._dedup:
            sorted_entries = deduplicate(sorted_entries, threshold=0.92)

        kept: list[ContextEntry] = []
        budget = self._max_tokens
        was_truncated = False
        for entry in sorted_entries:
            if entry.token_count <= budget:
                kept.append(entry)
                budget -= entry.token_count
            else:
                was_truncated = True

        if was_truncated:
            warnings.warn(
                f"Context exceeded max_tokens={self._max_tokens}. "
                "Lower-priority sources were dropped. "
                "Increase max_tokens if groundedness scores seem low.",
                ContextTooLargeWarning,
                stacklevel=2,
            )

        assembled = "\n\n---\n\n".join(e.content for e in kept)

        checksum = "sha256:" + hashlib.sha256(
            assembled.encode("utf-8")
        ).hexdigest()

        scrub_summary: dict[str, int] = {}
        pii_scrubbed = False
        for entry in kept:
            if entry.was_scrubbed:
                pii_scrubbed = True
                for k, v in entry.scrub_summary.items():
                    scrub_summary[k] = scrub_summary.get(k, 0) + v

        payload = ContextPayload(
            session_id=self._session_id,
            sources=kept,
            assembled_text=assembled,
            total_tokens=sum(e.token_count for e in kept),
            was_truncated=was_truncated,
            pii_scrubbed=pii_scrubbed,
            scrub_summary=scrub_summary,
            built_at=self._built_at,
            checksum=checksum,
        )

        audit.emit(
            "context_built",
            session_id=self._session_id,
            total_tokens=payload.total_tokens,
            was_truncated=was_truncated,
            sources_used=sorted({e.source for e in kept}),
            checksum=checksum,
        )
        return payload
