"""Structured, content-free audit logging (SOC II CC7).

Every ContextBuilder operation that touches content emits a structured
audit event - metadata only (entity-type counts, token counts, sources,
checksums), never the content itself.

Default destination is structured stderr (no file write in the OSS tier).
Enterprise deployments can route to a JSONL file with retention-based
rotation via :func:`configure_audit_log`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone, timedelta

_lock = threading.Lock()

_config: dict = {
    "destination": "stderr",   # "stderr" | "file" | "disabled"
    "path": None,
    "retention_days": 90,
}


def configure_audit_log(
    destination: str = "stderr",
    path: str | None = None,
    retention_days: int = 90,
) -> None:
    """Configure where scroot audit events are written.

    Args:
        destination: "stderr" (default - structured JSON lines on stderr),
            "file" (append to a JSONL file), or "disabled".
        path: JSONL file path, required when destination="file".
            ``~`` is expanded. Example: ``~/.scroot/audit.jsonl``.
        retention_days: For file destination, events older than this are
            pruned when the log is reconfigured or reopened. Default 90.

    Raises:
        ValueError: If destination is unknown, or destination="file"
            without a path.
    """
    if destination not in ("stderr", "file", "disabled"):
        raise ValueError(
            f"Unknown audit destination {destination!r}. "
            "Use 'stderr', 'file', or 'disabled'."
        )
    if destination == "file" and not path:
        raise ValueError("destination='file' requires a path.")

    resolved = os.path.expanduser(path) if path else None
    with _lock:
        _config["destination"] = destination
        _config["path"] = resolved
        _config["retention_days"] = retention_days
    if destination == "file":
        _rotate(resolved, retention_days)


def _rotate(path: str, retention_days: int) -> None:
    """Drop events older than retention_days. Never raises."""
    try:
        if not path or not os.path.exists(path):
            return
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat()
        kept = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("timestamp", "") >= cutoff:
                        kept.append(line)
                except json.JSONDecodeError:
                    continue
        with _lock:
            with open(path, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
    except OSError:
        pass


def emit(event: str, **fields) -> None:
    """Write one audit event. Metadata only - callers must never pass content.

    Failures are swallowed: audit logging must never crash the client's
    pipeline.

    Args:
        event: Event name, e.g. "context_entry_added", "context_built".
        **fields: JSON-serialisable metadata (counts, flags, checksums).
    """
    with _lock:
        destination = _config["destination"]
        path = _config["path"]
    if destination == "disabled":
        return

    from . import __version__
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
        "scroot_version": __version__,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return

    try:
        if destination == "file" and path:
            dir_name = os.path.dirname(os.path.abspath(path)) or "."
            os.makedirs(dir_name, exist_ok=True)
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        else:
            print(line, file=sys.stderr)
    except OSError:
        pass
