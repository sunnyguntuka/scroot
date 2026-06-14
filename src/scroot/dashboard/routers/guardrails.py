"""Guardrails router - /api/guardrails endpoints.

Surfaces the "loop closed" signal: which corrections have been included
in a GuardrailInjector.build_context() prompt, and how many times.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel


class GuardrailRecordStat(BaseModel):
    id: str
    guardrail_applied_count: int


class GuardrailStatsResponse(BaseModel):
    active_guardrails: int
    total_applications: int
    records: list[GuardrailRecordStat]


def guardrails_router(store):
    router = APIRouter()

    @router.get("/stats", response_model=GuardrailStatsResponse)
    def stats():
        records = store.get_all()
        active = [
            {"id": r.id, "guardrail_applied_count": getattr(r, "guardrail_applied_count", 0)}
            for r in records
            if getattr(r, "guardrail_applied_count", 0) > 0
        ]
        active.sort(key=lambda x: -x["guardrail_applied_count"])
        return {
            "active_guardrails": len(active),
            "total_applications": sum(r["guardrail_applied_count"] for r in active),
            "records": active,
        }

    return router
