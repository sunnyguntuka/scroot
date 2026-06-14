"""JSONL-based feedback store for corrections.

Security controls applied here:
  H-1: Path traversal validation on __init__ path argument.
  C-3: Optional Fernet encryption at rest; field masking for PII fields.
  C-4: delete(), purge(), max_records, ttl_days for data retention.
  H-7: Per-record SHA-256 hash + sequential record_number for tamper detection.

Each line is a self-contained correction record (plaintext or Fernet token).
Append-only writes; delete/purge rewrite atomically via a temp file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import warnings
from dataclasses import dataclass, asdict, fields


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectionRecord:
    """A single human-reviewed correction of an LLM response.

    Attributes:
        id: Unique identifier for this correction.
        timestamp: ISO 8601 UTC timestamp of when the correction was created.
        query: The original user query that produced the bad response.
        response: The original (incorrect) LLM-generated response.
        scores: Dict of metric scores from the Auditor at time of logging.
        flags: List of quality flags raised by the Auditor.
        correction: The correct answer that should have been given.
        reason: Human-readable explanation of why the original was wrong.
        context_used: List of source context strings used during scoring.
        corrected_by: Identity of the reviewer (not authenticated; for audit trail).
        status: Review lifecycle state.
            "pending"  - flagged, awaiting human or LLM-judge review.
            "reviewed" - correction written and verified.
            "rejected" - flagged response was actually acceptable; no fix needed.
            "applied"  - correction used as guardrail; improvement confirmed.
        corrected_response_iqs: IQS of the corrected response after re-scoring.
            None until the correction is re-evaluated with score().
        metadata: Optional free-form metadata dict.
        record_number: Sequential integer assigned on add() for gap detection (H-7).
        record_hash: SHA-256 hash of stable fields for tamper detection (H-7).
        session_id: ContextBuilder session id that produced the scored
            context, for multi-step trace reconstruction. Optional.
        context_checksum: SHA-256 checksum of the assembled context text
            (from ContextPayload). The context text itself is never stored
            here - only the integrity checksum.
        guardrail_applied_count: Number of times this correction has been
            included in a GuardrailInjector.build_context() prompt.
    """
    id: str
    timestamp: str
    query: str
    response: str
    scores: dict
    flags: list[str]
    correction: str
    reason: str
    context_used: list[str]
    corrected_by: str
    status: str = "pending"
    corrected_response_iqs: float | None = None
    metadata: dict | None = None
    record_number: int | None = None
    record_hash: str | None = None
    session_id: str | None = None
    context_checksum: str | None = None
    guardrail_applied_count: int = 0


_VALID_FIELDS = {f.name for f in fields(CorrectionRecord)}


# ---------------------------------------------------------------------------
# Record integrity helpers (H-7)
# ---------------------------------------------------------------------------

def _compute_record_hash(record: CorrectionRecord) -> str:
    """SHA-256 of stable identifying fields (first 16 hex chars)."""
    payload = f"{record.id}{record.timestamp}{record.query}{record.correction}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Thread-safety: per-path RLock (RLock allows re-entrant acquisition by
# the same thread, needed when add() → _prune_if_needed() → get_all())
# ---------------------------------------------------------------------------

_path_locks: dict[str, threading.RLock] = {}
_path_locks_mutex = threading.Lock()


def _get_thread_lock(path: str) -> threading.RLock:
    with _path_locks_mutex:
        if path not in _path_locks:
            _path_locks[path] = threading.RLock()
        return _path_locks[path]


# ---------------------------------------------------------------------------
# Unix advisory file locking (cross-process protection)
# ---------------------------------------------------------------------------

def _flock_file(f) -> None:
    if sys.platform != "win32":
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)


def _funlock_file(f) -> None:
    if sys.platform != "win32":
        import fcntl
        fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Path validation helper (H-1)
# ---------------------------------------------------------------------------

def _validate_store_path(path: str) -> str:
    """Resolve path, reject traversal attempts and non-.jsonl extensions.

    Returns the resolved absolute path.
    """
    normalized = os.path.normpath(path)
    parts = normalized.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError(
            f"Path traversal detected in store path: {path!r}"
        )
    if not normalized.endswith(".jsonl"):
        raise ValueError(
            f"Store path must end with '.jsonl', got: {path!r}"
        )
    return os.path.realpath(path)


# ---------------------------------------------------------------------------
# FeedbackStore
# ---------------------------------------------------------------------------

class FeedbackStore:
    """Append-only JSONL feedback store with optional encryption.

    Args:
        path: Path to the .jsonl file. Must end with '.jsonl'; relative paths
            containing '..' are rejected (H-1).
        encryption_key: Optional Fernet key (bytes or str) for encryption at
            rest (C-3). Generate one with ``cryptography.fernet.Fernet.generate_key()``.
            Requires ``pip install entail[security]``.
        field_mask: List of field names to replace with '[REDACTED]' before
            writing (C-3). Example: ``['query', 'context_used']``.
        max_records: If set, prune oldest records so count stays at or below
            this limit after each add() (C-4).
        ttl_days: If set, records older than this many days are pruned after
            each add() and get_all() (C-4).
    """

    def __init__(
        self,
        path: str = "feedback.jsonl",
        encryption_key=None,
        field_mask: list[str] | None = None,
        max_records: int | None = None,
        ttl_days: float | None = None,
    ):
        self.path = _validate_store_path(path)
        self.field_mask = field_mask or []
        self.max_records = max_records
        self.ttl_days = ttl_days
        self._warned_unencrypted = False

        self._fernet = None
        if encryption_key is not None:
            try:
                from cryptography.fernet import Fernet
            except ImportError as exc:
                raise ImportError(
                    "Encryption requires the cryptography package: "
                    "pip install 'scroot[security]'"
                ) from exc
            if isinstance(encryption_key, str):
                encryption_key = encryption_key.encode()
            self._fernet = Fernet(encryption_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_field_mask(self, data: dict) -> dict:
        if not self.field_mask:
            return data
        masked = dict(data)
        for field in self.field_mask:
            if field in masked:
                masked[field] = "[REDACTED]"
        return masked

    def _encode_line(self, data: dict) -> str:
        """Serialize, optionally encrypt, return a single line (no newline)."""
        line = json.dumps(data, ensure_ascii=False)
        if self._fernet is not None:
            line = self._fernet.encrypt(line.encode()).decode()
        return line

    def _decode_line(self, raw: str) -> dict | None:
        """Decrypt (if needed) and deserialize one line; return None on error."""
        try:
            if self._fernet is not None:
                raw = self._fernet.decrypt(raw.encode()).decode()
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return None
        except Exception:  # catches cryptography.fernet.InvalidToken and similar
            return None

    def _rewrite_unlocked(self, records: list[CorrectionRecord]) -> None:
        """Atomically rewrite the store. Caller must hold the path lock."""
        dir_name = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl.tmp", dir=dir_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                for record in records:
                    data = self._apply_field_mask(asdict(record))
                    tmp.write(self._encode_line(data) + "\n")
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _get_all_raw(self) -> list[CorrectionRecord]:
        """Read records without acquiring the lock (internal use)."""
        records = []
        if not os.path.exists(self.path):
            return records
        with open(self.path, "r", encoding="utf-8") as f:
            _flock_file(f)
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = self._decode_line(line)
                    if data is None:
                        continue
                    filtered = {k: v for k, v in data.items() if k in _VALID_FIELDS}
                    records.append(CorrectionRecord(**filtered))
            finally:
                _funlock_file(f)
        return records

    def _cutoff_timestamp(self) -> str | None:
        if self.ttl_days is None:
            return None
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.ttl_days)
        return cutoff.isoformat()

    def _prune_if_needed(self) -> None:
        """Enforce max_records and ttl_days after an add()."""
        if self.max_records is None and self.ttl_days is None:
            return
        lock = _get_thread_lock(self.path)
        with lock:
            records = self._get_all_raw()
            cutoff = self._cutoff_timestamp()
            if cutoff is not None:
                records = [r for r in records if r.timestamp >= cutoff]
            if self.max_records is not None and len(records) > self.max_records:
                records = records[-self.max_records:]
            self._rewrite_unlocked(records)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: CorrectionRecord) -> None:
        """Append a correction record to the store.

        Args:
            record: CorrectionRecord to persist. record_number and
                record_hash are assigned automatically if not set.
        """
        if not self._warned_unencrypted and self._fernet is None:
            warnings.warn(
                "FeedbackStore writing unencrypted records. "
                "Set encryption_key for PII protection.",
                stacklevel=2,
            )
            self._warned_unencrypted = True

        # Assign sequential number and integrity hash before writing.
        if record.record_number is None:
            record.record_number = self.count() + 1
        if record.record_hash is None:
            record.record_hash = _compute_record_hash(record)

        data = self._apply_field_mask(asdict(record))

        lock = _get_thread_lock(self.path)
        with lock:
            dir_name = os.path.dirname(os.path.abspath(self.path)) or "."
            os.makedirs(dir_name, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                _flock_file(f)
                try:
                    f.write(self._encode_line(data) + "\n")
                finally:
                    _funlock_file(f)

        self._prune_if_needed()

    def get_all(self) -> list[CorrectionRecord]:
        """Read all correction records, applying TTL filtering."""
        lock = _get_thread_lock(self.path)
        with lock:
            records = self._get_all_raw()
        cutoff = self._cutoff_timestamp()
        if cutoff is not None:
            records = [r for r in records if r.timestamp >= cutoff]
        return records

    def get_recent(self, n: int = 10) -> list[CorrectionRecord]:
        """Get the N most recent corrections.

        Args:
            n: Maximum number of records to return (default 10).

        Returns:
            The n most recent CorrectionRecords, oldest first.
        """
        return self.get_all()[-n:]

    def delete(self, record_id: str) -> int:
        """Delete all records with the given id.

        Args:
            record_id: The id field to match for deletion.

        Returns:
            Number of records deleted (0 if id not found).
        """
        lock = _get_thread_lock(self.path)
        with lock:
            records = self._get_all_raw()
            remaining = [r for r in records if r.id != record_id]
            deleted = len(records) - len(remaining)
            if deleted > 0:
                self._rewrite_unlocked(remaining)
        return deleted

    def purge(self, before_date: str) -> int:
        """Remove records with timestamp < before_date (ISO 8601 string).

        Args:
            before_date: ISO 8601 datetime string. Records with an earlier
                timestamp are removed.

        Returns:
            Number of records removed.
        """
        lock = _get_thread_lock(self.path)
        with lock:
            records = self._get_all_raw()
            remaining = [r for r in records if r.timestamp >= before_date]
            removed = len(records) - len(remaining)
            if removed > 0:
                self._rewrite_unlocked(remaining)
        return removed

    def validate_integrity(self) -> list[str]:
        """Check record hashes and sequence numbers for tampering.

        Returns a list of issue descriptions. An empty list means no
        integrity problems were detected.
        """
        records = self.get_all()
        issues = []

        numbered = [r for r in records if r.record_number is not None]
        if numbered:
            numbers = sorted(r.record_number for r in numbered)
            for expected, actual in enumerate(numbers, start=numbers[0]):
                if actual != expected:
                    issues.append(
                        f"Record number gap: expected {expected}, found {actual}"
                    )

        for r in records:
            if r.record_hash is None:
                continue
            expected_hash = _compute_record_hash(r)
            if r.record_hash != expected_hash:
                issues.append(
                    f"Hash mismatch on record {r.id!r}: "
                    f"stored={r.record_hash!r}, computed={expected_hash!r}"
                )

        return issues

    def search(
        self,
        query: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        top_k: int = 5,
    ) -> list[CorrectionRecord]:
        """Find corrections most relevant to a query using embedding similarity.

        Args:
            query: The current user query.
            embedding_model: Sentence-transformers model name.
            device: "cpu" or "cuda" - must match the device used by the
                Auditor to avoid loading a second model instance (M-4).
            top_k: Number of results to return.

        Returns:
            Most relevant correction records, sorted by similarity.
        """
        import numpy as np
        from ..models import get_embedding_model

        records = self.get_all()
        if not records:
            return []

        model = get_embedding_model(embedding_model, device=device)
        query_emb = model.encode(query, convert_to_numpy=True)
        record_queries = [r.query for r in records]
        record_embs = model.encode(record_queries, convert_to_numpy=True)

        similarities = np.dot(record_embs, query_emb) / (
            np.linalg.norm(record_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [records[i] for i in top_indices if similarities[i] > 0.3]

    def count(self) -> int:
        """Count total records (does not apply TTL filter)."""
        if not os.path.exists(self.path):
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # ------------------------------------------------------------------
    # Review lifecycle
    # ------------------------------------------------------------------

    def get_pending(self) -> list[CorrectionRecord]:
        """Return records with status='pending' awaiting review.

        These are responses scroot flagged as low-quality but have not yet
        been reviewed by a human or LLM-judge to produce a correction.

        Returns:
            List of CorrectionRecords with status == 'pending', oldest first.
        """
        return [r for r in self.get_all() if getattr(r, "status", "pending") == "pending"]

    def mark_reviewed(
        self,
        record_id: str,
        correction: str,
        reason: str | None = None,
        corrected_by: str = "human",
        status: str = "reviewed",
        corrected_response_iqs: float | None = None,
    ) -> bool:
        """Update a record with a correction and advance its review status.

        Atomically rewrites the store so the updated record is persisted
        safely even if a crash occurs mid-write.

        Args:
            record_id: The id of the record to update.
            correction: The correct response that should have been generated.
            reason: Updated explanation of why the original was wrong.
                If None, the existing reason is kept.
            corrected_by: Identity of the reviewer ('human', 'gpt-4o', etc.).
            status: New lifecycle state. One of:
                'reviewed'  - correction written and verified (default).
                'rejected'  - original response was acceptable; no fix needed.
                'applied'   - correction confirmed to improve IQS.
            corrected_response_iqs: IQS after re-scoring the correction.
                Pass the result of auditor.score(correction, ...).iqs here.

        Returns:
            True if the record was found and updated, False if not found.
        """
        lock = _get_thread_lock(self.path)
        with lock:
            records = self._get_all_raw()
            found = False
            for r in records:
                if r.id == record_id:
                    r.correction = correction
                    if reason is not None:
                        r.reason = reason
                    r.corrected_by = corrected_by
                    r.status = status
                    if corrected_response_iqs is not None:
                        r.corrected_response_iqs = corrected_response_iqs
                    found = True
                    break
            if found:
                self._rewrite_unlocked(records)
        return found

    def increment_guardrail_count(self, record_ids: list[str]) -> None:
        """Increment guardrail_applied_count for the given record ids.

        Called by GuardrailInjector.build_context() to track which
        corrections were actually included in a generated prompt
        ("loop closed" tracking).

        Args:
            record_ids: ids of records that were included in a guardrail
                prompt. Unknown ids are ignored.
        """
        if not record_ids:
            return
        ids = set(record_ids)
        lock = _get_thread_lock(self.path)
        with lock:
            records = self._get_all_raw()
            changed = False
            for r in records:
                if r.id in ids:
                    r.guardrail_applied_count = getattr(r, "guardrail_applied_count", 0) + 1
                    changed = True
            if changed:
                self._rewrite_unlocked(records)

    def get_by_status(self, status: str) -> list[CorrectionRecord]:
        """Return all records with the given status.

        Args:
            status: One of 'pending', 'reviewed', 'rejected', 'applied'.

        Returns:
            List of matching CorrectionRecords, oldest first.
        """
        return [r for r in self.get_all() if getattr(r, "status", "pending") == status]

    # ------------------------------------------------------------------
    # Fine-tuning export
    # ------------------------------------------------------------------

    def export_for_finetuning(
        self,
        output_path: str,
        fmt: str = "openai",
        status_filter: list[str] | None = None,
        system_prompt: str = "You are a helpful assistant. Answer questions accurately based on the provided context.",
    ) -> int:
        """Export correction records as supervised fine-tuning training pairs.

        Exports only records that have a non-empty correction field and pass
        the status filter. Records with status='rejected' are excluded by
        default (they indicate the original response was acceptable).

        Args:
            output_path: Path to write the JSONL output file.
            fmt: Output format. One of:
                'openai'  - OpenAI chat fine-tuning format (default).
                            {"messages": [{"role": "system",...},
                                          {"role": "user",...},
                                          {"role": "assistant",...}]}
                'alpaca'  - Alpaca instruction format.
                            {"instruction":..., "input":..., "output":...}
                'simple'  - Minimal prompt/completion pairs.
                            {"prompt":..., "completion":...}
            status_filter: Only export records with these statuses.
                Defaults to ['reviewed', 'applied'].
            system_prompt: System prompt injected in 'openai' format.

        Returns:
            Number of records exported.
        """
        if status_filter is None:
            status_filter = ["reviewed", "applied"]

        records = [
            r for r in self.get_all()
            if getattr(r, "status", "pending") in status_filter
            and r.correction.strip()
        ]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        exported = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                context_block = ""
                if r.context_used:
                    ctx = "\n".join(f"- {c}" for c in r.context_used)
                    context_block = f"\n\nContext:\n{ctx}"

                user_message = f"{r.query}{context_block}"

                if fmt == "openai":
                    entry = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": r.correction},
                        ]
                    }
                elif fmt == "alpaca":
                    entry = {
                        "instruction": system_prompt,
                        "input": user_message,
                        "output": r.correction,
                    }
                elif fmt == "simple":
                    prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
                    entry = {"prompt": prompt, "completion": r.correction}
                else:
                    raise ValueError(f"Unknown format {fmt!r}. Use 'openai', 'alpaca', or 'simple'.")

                # Attach metadata useful for downstream filtering
                entry["_scroot_meta"] = {
                    "id": r.id,
                    "original_iqs": r.scores.get("iqs"),
                    "corrected_response_iqs": r.corrected_response_iqs,
                    "flags": r.flags,
                    "corrected_by": r.corrected_by,
                    "status": getattr(r, "status", "reviewed"),
                }

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                exported += 1

        return exported
