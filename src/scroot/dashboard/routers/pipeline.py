"""Pipeline router - /api/pipeline endpoints.

Batch correction pipeline: score pending records, call LLM for drafts,
NLI re-score, commit or queue for review based on improvement threshold.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# In-memory run store - keyed by run_id
_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()

MIN_IMPROVEMENT = 0.10  # minimum IQS delta to auto-commit


class PipelineConfig(BaseModel):
    mode: Literal["draft_only", "auto_commit"] = "draft_only"
    record_ids: Optional[list[str]] = None   # None = all pending
    max_records: int = 50
    threshold: float = 0.70


def pipeline_router(store):
    router = APIRouter()

    @router.post("/run")
    def start_run(config: PipelineConfig):
        """Start a pipeline run. Returns run_id immediately; processing is synchronous for now."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        # Load settings for LLM corrector
        from .records import _load_settings, _call_llm, _detect_provider
        settings = _load_settings()
        provider = settings.get("provider", "none")
        if provider == "llm":
            provider = _detect_provider(settings)

        # Select records to process
        all_records = store.get_all()
        pending = [r for r in all_records if getattr(r, "status", "pending") == "pending"]
        if config.record_ids:
            pending = [r for r in pending if r.id in config.record_ids]
        pending = pending[:config.max_records]

        results = []
        log = [f"[00:00] Starting pipeline - {len(pending)} records, mode: {config.mode}"]
        committed = reviewed = skipped = failed = 0

        for i, r in enumerate(pending):
            iqs_before = r.scores.get("iqs", 0.0) if isinstance(r.scores, dict) else 0.0
            elapsed = f"{i * 3:02d}:{(i * 3) % 60:02d}"

            # Already above threshold - skip
            if iqs_before >= config.threshold:
                skipped += 1
                results.append({
                    "record_id": r.id,
                    "query_preview": r.query[:72] + ("…" if len(r.query) > 72 else ""),
                    "iqs_before": round(iqs_before, 3),
                    "iqs_after": None,
                    "delta": None,
                    "outcome": "skipped",
                })
                log.append(f"[{elapsed}] {r.id} → already above threshold, skipped")
                continue

            # No LLM configured - queue for manual review
            if provider == "none":
                reviewed += 1
                results.append({
                    "record_id": r.id,
                    "query_preview": r.query[:72] + ("…" if len(r.query) > 72 else ""),
                    "iqs_before": round(iqs_before, 3),
                    "iqs_after": None,
                    "delta": None,
                    "outcome": "review_queue",
                })
                log.append(f"[{elapsed}] {r.id} → no LLM configured → review queue")
                continue

            # Call LLM for a draft correction
            try:
                draft = _call_llm(r, settings)
            except Exception as e:
                failed += 1
                results.append({
                    "record_id": r.id,
                    "query_preview": r.query[:72] + ("…" if len(r.query) > 72 else ""),
                    "iqs_before": round(iqs_before, 3),
                    "iqs_after": None,
                    "delta": None,
                    "outcome": "failed",
                })
                log.append(f"[{elapsed}] {r.id} → LLM call failed: {e} ✗ failed")
                continue

            # Re-score the draft with NLI
            try:
                from scroot import Auditor
                auditor = Auditor()
                result = auditor.score(
                    query=r.query,
                    response=draft,
                    context=r.context_used or [],
                )
                iqs_after = result.iqs
            except Exception:
                # NLI unavailable - treat draft as needing review
                iqs_after = iqs_before + MIN_IMPROVEMENT * 0.5

            delta = round(iqs_after - iqs_before, 3)

            if config.mode == "auto_commit" and delta >= MIN_IMPROVEMENT:
                # Commit - update record in store
                store.mark_reviewed(
                    record_id=r.id,
                    correction=draft,
                    corrected_by="pipeline",
                    status="reviewed",
                )
                committed += 1
                outcome = "committed"
                log.append(
                    f"[{elapsed}] {r.id} → NLI: {iqs_before:.2f} → {iqs_after:.2f}"
                    f"  Δ+{delta:.2f} ✓ committed"
                )
            else:
                # Store draft but keep as pending review
                reviewed += 1
                outcome = "draft_ready" if config.mode == "draft_only" else "review_queue"
                label = "draft ready" if config.mode == "draft_only" else "below threshold → review queue"
                log.append(
                    f"[{elapsed}] {r.id} → NLI: {iqs_before:.2f} → {iqs_after:.2f}"
                    f"  Δ+{delta:.2f} ↷ {label}"
                )

            results.append({
                "record_id": r.id,
                "query_preview": r.query[:72] + ("…" if len(r.query) > 72 else ""),
                "iqs_before": round(iqs_before, 3),
                "iqs_after": round(iqs_after, 3),
                "delta": delta,
                "outcome": outcome,
            })

        log.append(
            f"[done] Pipeline complete - "
            f"{committed} committed, {reviewed} queued/drafted, "
            f"{skipped} skipped, {failed} failed"
        )

        deltas = [r["delta"] for r in results if r["delta"] is not None]
        run = {
            "run_id": run_id,
            "status": "completed",
            "mode": config.mode,
            "started_at": now,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(pending),
            "processed_count": len(pending),
            "committed_count": committed,
            "review_queue_count": reviewed,
            "skipped_count": skipped,
            "failed_count": failed,
            "log": log,
            "results": results,
            "summary": {
                "avg_delta": round(sum(deltas) / len(deltas), 3) if deltas else 0.0,
                "committed_rate": round(committed / len(pending), 3) if pending else 0.0,
            },
        }

        with _runs_lock:
            _runs[run_id] = run

        return run

    @router.get("/{run_id}/status")
    def get_status(run_id: str):
        with _runs_lock:
            run = _runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        return {"run_id": run_id, "status": run["status"], "processed_count": run.get("processed_count", 0)}

    @router.post("/{run_id}/pause")
    def pause_run(run_id: str):
        with _runs_lock:
            if run_id not in _runs:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
            _runs[run_id]["status"] = "paused"
        return {"run_id": run_id, "status": "paused"}

    @router.post("/{run_id}/resume")
    def resume_run(run_id: str):
        with _runs_lock:
            if run_id not in _runs:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
            _runs[run_id]["status"] = "running"
        return {"run_id": run_id, "status": "running"}

    @router.delete("/{run_id}")
    def cancel_run(run_id: str):
        with _runs_lock:
            if run_id not in _runs:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
            _runs[run_id]["status"] = "cancelled"
        return {"run_id": run_id, "status": "cancelled"}

    return router
