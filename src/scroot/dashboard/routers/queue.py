"""Queue router - /api/queue endpoints."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel


class QueueItem(BaseModel):
    id: str
    agent_id: str
    query: str
    response: str
    iqs: float
    flags: list[str]
    status: Literal["pending", "claimed", "reviewed", "rejected", "applied"]
    created_at: str
    claimed_at: Optional[str] = None
    iqs_metric_count: int = 5
    session_id: Optional[str] = None
    context_checksum: Optional[str] = None


class QueueResponse(BaseModel):
    records: list[QueueItem]
    total: int
    page: int


class StatsResponse(BaseModel):
    pending: int
    reviewed_today: int
    avg_iqs: float
    oldest_pending_hours: float


# Atomic claim registry - single-session open-source tier
_claims: dict[str, dict] = {}
_claims_lock = threading.Lock()


def queue_router(store):
    router = APIRouter()

    @router.get("", response_model=QueueResponse)
    def list_queue(
        status: str = Query("all"),
        flag: Optional[str] = Query(None),
        agent: Optional[str] = Query(None),
        min_iqs: Optional[float] = Query(None),
        max_iqs: Optional[float] = Query(None),
        threshold: float = Query(0.70),
        sort: str = Query("created_desc"),
        page: int = Query(1),
        limit: int = Query(50),
        search: Optional[str] = Query(None),
    ):
        records = store.get_all()

        # IQS status filter (pass / warn / fail) - quality-based
        if status in ("pass", "warn", "fail"):
            warn_floor = threshold * 0.7
            def iqs_status(r):
                iqs = r.scores.get("iqs", 0) if isinstance(r.scores, dict) else 0
                if iqs >= threshold:
                    return "pass"
                if iqs >= warn_floor:
                    return "warn"
                return "fail"
            records = [r for r in records if iqs_status(r) == status]
        elif status != "all":
            # Workflow status filter (pending / reviewed / rejected)
            records = [r for r in records if getattr(r, "status", "pending") == status]

        # Text search
        if search:
            q = search.lower()
            records = [r for r in records if q in r.query.lower()]

        # Filter by flag
        if flag:
            records = [r for r in records if flag in (r.flags or [])]

        # Filter by IQS range
        if min_iqs is not None:
            records = [r for r in records if r.scores.get("iqs", 0) >= min_iqs]
        if max_iqs is not None:
            records = [r for r in records if r.scores.get("iqs", 1) <= max_iqs]

        # Sort
        reverse = sort.endswith("_desc")
        key_map = {
            "iqs_asc":      lambda r: r.scores.get("iqs", 0),
            "iqs_desc":     lambda r: r.scores.get("iqs", 0),
            "created_asc":  lambda r: r.timestamp,
            "created_desc": lambda r: r.timestamp,
            "newest":       lambda r: r.timestamp,
            "oldest":       lambda r: r.timestamp,
        }
        reverse = sort in ("iqs_desc", "created_desc", "newest")
        sort_key = key_map.get(sort, lambda r: r.timestamp)
        records = sorted(records, key=sort_key, reverse=reverse)

        total = len(records)
        start = (page - 1) * limit
        page_records = records[start: start + limit]

        items = []
        for r in page_records:
            sc = r.scores if isinstance(r.scores, dict) else {}
            iqs = sc.get("iqs", 0.0)
            metric_count = sc.get(
                "iqs_metric_count", 5 if sc.get("groundedness") is not None else 4
            )
            claim = _claims.get(r.id)
            items.append(QueueItem(
                id=r.id,
                agent_id=r.corrected_by or "unknown",
                query=r.query[:120],
                response=r.response[:200],
                iqs=iqs,
                flags=r.flags or [],
                status=getattr(r, "status", "pending"),
                created_at=r.timestamp,
                claimed_at=claim.get("claimed_at") if claim else None,
                session_id=getattr(r, "session_id", None),
                context_checksum=getattr(r, "context_checksum", None),
                iqs_metric_count=metric_count,
            ))

        return QueueResponse(records=items, total=total, page=page)

    @router.post("/claim/{record_id}")
    def claim_record(record_id: str):
        """Atomic claim - 409 if already claimed by another session."""
        with _claims_lock:
            if record_id in _claims:
                raise HTTPException(
                    status_code=409,
                    detail=f"Record {record_id} is already claimed",
                )
            now = datetime.now(timezone.utc).isoformat()
            _claims[record_id] = {"claimed_at": now}
            return {"record_id": record_id, "claimed_at": now, "status": "claimed"}

    @router.delete("/claim/{record_id}")
    def unclaim_record(record_id: str):
        """Release a claim when reviewer navigates away."""
        with _claims_lock:
            _claims.pop(record_id, None)
        return {"record_id": record_id, "status": "released"}

    @router.get("/stats", response_model=StatsResponse)
    def queue_stats():
        all_records = store.get_all()
        pending = [r for r in all_records if getattr(r, "status", "pending") == "pending"]

        today = datetime.now(timezone.utc).date().isoformat()
        reviewed_today = sum(
            1 for r in all_records
            if getattr(r, "status", "pending") == "reviewed"
            and r.timestamp[:10] == today
        )

        iqs_vals = [r.scores.get("iqs", 0) for r in all_records if isinstance(r.scores, dict)]
        avg_iqs = sum(iqs_vals) / len(iqs_vals) if iqs_vals else 0.0

        oldest_hours = 0.0
        if pending:
            oldest_ts = min(r.timestamp for r in pending)
            try:
                dt = datetime.fromisoformat(oldest_ts.replace("Z", "+00:00"))
                delta = datetime.now(timezone.utc) - dt
                oldest_hours = delta.total_seconds() / 3600
            except (ValueError, AttributeError):
                pass

        return StatsResponse(
            pending=len(pending),
            reviewed_today=reviewed_today,
            avg_iqs=round(avg_iqs, 3),
            oldest_pending_hours=round(oldest_hours, 1),
        )

    return router
