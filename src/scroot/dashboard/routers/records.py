"""Records router - /api/records/:id endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class ReviewBody(BaseModel):
    correction: str
    category: str = "manual"
    notes: Optional[str] = None


class RejectBody(BaseModel):
    reason: str


# The 5 IQS metrics - used to whitelist `metrics` so derived to_dict() fields
# (weakest_metric, score_variance, etc.) don't leak in as fake metric scores.
_METRIC_KEYS = {"groundedness", "completeness", "relevance", "consistency", "confidence"}


def _record_to_dict(r) -> dict:
    """Serialize a CorrectionRecord to the shape the frontend expects."""
    scores = r.scores if isinstance(r.scores, dict) else {}
    iqs = scores.get("iqs", 0.0)
    metrics = {k: v for k, v in scores.items() if k in _METRIC_KEYS}
    return {
        "id": r.id,
        "timestamp": r.timestamp,
        "created_at": r.timestamp,
        "query": r.query,
        "response": r.response,
        "context": "\n".join(r.context_used or []),
        "correction": r.correction,
        "rejection_reason": r.reason,
        "corrected_by": r.corrected_by,
        "status": getattr(r, "status", "pending"),
        "iqs": iqs,
        "metrics": metrics,
        "flags": r.flags or [],
        "corrected_response_iqs": getattr(r, "corrected_response_iqs", None),
        "agent_id": r.corrected_by or None,
        "model": None,
        "weakest_metric": scores.get("weakest_metric"),
        "score_variance": scores.get("score_variance"),
        "iqs_explanation": scores.get("iqs_explanation"),
        "metric_explanations": scores.get("metric_explanations") or {},
        "guardrail_applied_count": getattr(r, "guardrail_applied_count", 0),
        "evidence_map": scores.get("evidence_map"),
        # IQS transparency: whether groundedness was scored and how many metrics
        # contributed (defaults derived for older records without these keys).
        "context_used": scores.get("context_used", scores.get("groundedness") is not None),
        "iqs_metric_count": scores.get(
            "iqs_metric_count", 5 if scores.get("groundedness") is not None else 4
        ),
        "effective_weights": scores.get("effective_weights"),
    }


def records_router(store):
    router = APIRouter()

    @router.get("/{record_id}")
    def get_record(record_id: str):
        records = store.get_all()
        match = next((r for r in records if r.id == record_id), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
        return _record_to_dict(match)

    @router.post("/{record_id}/review")
    def submit_review(record_id: str, body: ReviewBody):
        if not body.correction.strip():
            raise HTTPException(status_code=422, detail="Correction cannot be empty")

        ok = store.mark_reviewed(
            record_id=record_id,
            correction=body.correction,
            corrected_by="reviewer",
            status="reviewed",
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")

        # Release claim
        from .queue import _claims, _claims_lock
        with _claims_lock:
            _claims.pop(record_id, None)

        # Return updated record so frontend can setRecord() directly
        records = store.get_all()
        updated = next((r for r in records if r.id == record_id), None)
        if updated:
            return _record_to_dict(updated)
        return {"record_id": record_id, "status": "reviewed"}

    @router.post("/{record_id}/reject")
    def reject_record(record_id: str, body: RejectBody):
        ok = store.mark_reviewed(
            record_id=record_id,
            correction="",
            reason=body.reason,
            corrected_by="reviewer",
            status="rejected",
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")

        from .queue import _claims, _claims_lock
        with _claims_lock:
            _claims.pop(record_id, None)

        records = store.get_all()
        updated = next((r for r in records if r.id == record_id), None)
        if updated:
            return _record_to_dict(updated)
        return {"record_id": record_id, "status": "rejected"}

    @router.delete("/{record_id}/correction")
    def delete_correction(record_id: str):
        """Reset a record to pending - undoes a correction or rejection."""
        ok = store.mark_reviewed(
            record_id=record_id,
            correction="",
            reason="",
            corrected_by=None,
            status="pending",
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")

        records = store.get_all()
        updated = next((r for r in records if r.id == record_id), None)
        if updated:
            return _record_to_dict(updated)
        return {"record_id": record_id, "status": "pending"}

    @router.post("/{record_id}/generate-correction")
    async def generate_correction(record_id: str):
        """
        Call the configured LLM and return a draft correction as JSON.
        NEVER auto-populates the frontend - user must click Generate.
        """
        records = store.get_all()
        match = next((r for r in records if r.id == record_id), None)
        if not match:
            raise HTTPException(status_code=404, detail="Record not found")

        settings = _load_settings()
        provider = settings.get("provider", "none")

        if provider == "none":
            raise HTTPException(status_code=400, detail="No LLM corrector configured. Set one in Settings.")

        try:
            draft = _call_llm(match, settings)
            return {"draft": draft}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    return router


def _load_settings() -> dict:
    """Load persisted LLM judge settings."""
    import json
    import os
    settings_path = os.path.join(os.getcwd(), ".scroot_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            return json.load(f)
    return {"provider": "none"}


def _detect_provider(settings: dict) -> str:
    """Infer the actual API provider from base_url and model name."""
    base_url = (settings.get("base_url") or "").lower()
    model = (settings.get("model") or "").lower()
    if "localhost:11434" in base_url or "ollama" in base_url:
        return "ollama"
    if "anthropic" in base_url or model.startswith("claude"):
        return "anthropic"
    if "groq" in base_url:
        return "groq"
    if "openrouter" in base_url:
        return "openrouter"
    return "openai"


def _call_llm(record, settings: dict) -> str:
    """Call the configured LLM provider and return a correction draft."""
    provider = settings.get("provider", "none")
    if provider == "llm":
        provider = _detect_provider(settings)

    model = settings.get("model", "")
    api_key = settings.get("api_key", "")
    base_url = settings.get("base_url", "") or None

    # Fall back to env var if direct key not stored
    api_key_env = settings.get("api_key_env_var", "")
    if not api_key and api_key_env:
        import os
        api_key = os.environ.get(api_key_env, "")

    # M-2: refuse to send the API key to an unvetted/internal endpoint.
    from scroot.dashboard.security import validate_base_url
    validate_base_url(base_url)

    context_text = "\n".join(record.context_used or [])
    prompt = (
        f"Query: {record.query}\n"
        f"Context: {context_text}\n"
        f"Problematic response: {record.response}\n"
        f"Flags: {', '.join(record.flags or [])}\n\n"
        f"Write a corrected, grounded response:"
    )

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        msg = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif provider in ("openai", "groq", "openrouter"):
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content

    elif provider == "ollama":
        import requests
        url = (base_url or "http://localhost:11434") + "/api/generate"
        resp = requests.post(
            url,
            json={"model": model or "llama3.2", "prompt": prompt, "stream": False},
            timeout=60,
        )
        return resp.json().get("response", "")

    return "No LLM provider configured."
