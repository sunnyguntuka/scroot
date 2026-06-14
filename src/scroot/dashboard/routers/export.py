"""Export router - /api/export endpoints."""
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class ExportFilters(BaseModel):
    status: list[str] = ["reviewed", "applied"]
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_iqs: Optional[float] = None
    max_iqs: Optional[float] = None
    flags: list[str] = []
    agents: list[str] = []


class ExportRequest(BaseModel):
    filters: ExportFilters
    format: Literal["jsonl", "csv", "parquet"] = "jsonl"
    system_prompt: str = (
        "You are a helpful assistant. Answer questions accurately "
        "based on the provided context."
    )


class S3PushRequest(BaseModel):
    filters: ExportFilters
    format: Literal["jsonl", "csv", "parquet"] = "jsonl"
    bucket: str
    prefix: str = "scroot-exports/"


def export_router(store):
    router = APIRouter()

    def _apply_filters(records, filters: ExportFilters):
        out = []
        for r in records:
            status = getattr(r, "status", "pending")
            if filters.status and status not in filters.status:
                continue
            if filters.date_from and r.timestamp[:10] < filters.date_from:
                continue
            if filters.date_to and r.timestamp[:10] > filters.date_to:
                continue
            iqs = r.scores.get("iqs", 0) if isinstance(r.scores, dict) else 0
            if filters.min_iqs is not None and iqs < filters.min_iqs:
                continue
            if filters.max_iqs is not None and iqs > filters.max_iqs:
                continue
            if filters.flags:
                if not any(f in (r.flags or []) for f in filters.flags):
                    continue
            if not r.correction.strip():
                continue
            out.append(r)
        return out

    @router.post("/preview")
    def preview(body: ExportRequest):
        records = store.get_all()
        matched = _apply_filters(records, body.filters)
        sample = []
        for r in matched[:3]:
            sample.append({
                "id": r.id,
                "agent_id": r.corrected_by or "",
                "iqs": r.scores.get("iqs", 0) if isinstance(r.scores, dict) else 0,
                "flags": r.flags or [],
                "status": getattr(r, "status", "pending"),
            })
        corrected = [r for r in matched if r.correction and r.correction.strip()]
        agents = sorted({r.corrected_by for r in records if r.corrected_by})
        return {
            "count": len(matched),
            "corrected_count": len(corrected),
            "agents": agents,
            "sample": sample,
        }

    @router.post("/download")
    def download(body: ExportRequest):
        records = store.get_all()
        matched = _apply_filters(records, body.filters)

        if body.format == "jsonl":
            lines = []
            for r in matched:
                ctx = "\n".join(r.context_used or [])
                entry = {
                    "messages": [
                        {"role": "system", "content": body.system_prompt},
                        {"role": "user", "content": f"{r.query}\n\nContext:\n{ctx}"},
                        {"role": "assistant", "content": r.correction},
                    ],
                    "_meta": {
                        "id": r.id,
                        "original_iqs": r.scores.get("iqs") if isinstance(r.scores, dict) else None,
                        "flags": r.flags,
                        "corrected_by": r.corrected_by,
                    },
                }
                lines.append(json.dumps(entry, ensure_ascii=False))
            content = "\n".join(lines).encode("utf-8")
            media_type = "application/jsonl"
            filename = f"scroot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        elif body.format == "csv":
            import csv
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["id", "query", "context", "bad_response", "correction", "flags", "original_iqs"])
            for r in matched:
                writer.writerow([
                    r.id, r.query,
                    "; ".join(r.context_used or []),
                    r.response, r.correction,
                    "|".join(r.flags or []),
                    r.scores.get("iqs") if isinstance(r.scores, dict) else "",
                ])
            content = buf.getvalue().encode("utf-8")
            media_type = "text/csv"
            filename = f"scroot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        else:
            raise ValueError(f"Unsupported format: {body.format}")

        return StreamingResponse(
            io.BytesIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.post("/push-s3")
    def push_s3(body: S3PushRequest):
        """Push export to S3 - single destination in open-source tier."""
        import uuid
        job_id = str(uuid.uuid4())[:8]
        # In open-source tier, this queues but doesn't schedule multi-destination
        return {"status": "queued", "job_id": job_id,
                "note": "Multi-destination scheduling available in Scroot Enterprise"}

    return router
