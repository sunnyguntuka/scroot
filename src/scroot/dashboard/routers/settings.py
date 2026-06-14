"""Settings router - /api/settings endpoints."""
from __future__ import annotations

import json
import os
import time

from fastapi import APIRouter, HTTPException

from scroot.config.corrector import (
    APIConfig,
    CorrectorConfig,
    LocalConfig,
    default_config_path,
)
from scroot.corrector.models import MODEL_REGISTRY, get_model_path
from scroot.dashboard.security import mask_api_key, validate_base_url

_SETTINGS_FILE = os.path.join(os.getcwd(), ".scroot_settings.json")

DEFAULT_WEIGHTS = {
    "groundedness": 0.35,
    "completeness": 0.25,
    "relevance": 0.20,
    "consistency": 0.15,
    "confidence": 0.05,
}

DEFAULT_CONFIG: dict = {
    "iqs_threshold": 0.70,
    "metric_weights": DEFAULT_WEIGHTS,
    "provider": "none",
    "model": "",
    "api_key": "",
    "base_url": "",
    "trigger_mode": "manual",
}


def _load() -> dict:
    if os.path.exists(_SETTINGS_FILE):
        with open(_SETTINGS_FILE, encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()


def _save(config: dict) -> None:
    # M-1: write key-bearing settings as owner-only (0600) so other local
    # accounts can't read the stored API key.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(_SETTINGS_FILE, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    try:
        os.chmod(_SETTINGS_FILE, 0o600)  # tighten if the file pre-existed
    except OSError:
        pass


def _store_info(store) -> dict:
    """Return record count and human-readable store size."""
    if store is None:
        return {"record_count": 0, "store_size": "—", "store_path": "~/.scroot/feedback.jsonl"}
    try:
        records = store.get_all()
        count = len(records)
        path = getattr(store, "_path", str(getattr(store, "path", "~/.scroot/feedback.jsonl")))
        size = "—"
        if os.path.exists(path):
            b = os.path.getsize(path)
            if b < 1024:
                size = f"{b} B"
            elif b < 1024 ** 2:
                size = f"{b / 1024:.1f} KB"
            else:
                size = f"{b / 1024 ** 2:.1f} MB"
        return {"record_count": count, "store_size": size, "store_path": path}
    except (OSError, AttributeError):
        return {"record_count": 0, "store_size": "—", "store_path": "~/.scroot/feedback.jsonl"}


def settings_router(store=None):
    router = APIRouter()

    # ─── Unified settings (used by Settings page) ─────────────────────

    def _corrector_state() -> dict:
        """Build the corrector sub-object returned in GET /settings."""
        cc = CorrectorConfig.load(default_config_path())
        spec = MODEL_REGISTRY.get(cc.local.model_id)
        model_path = get_model_path(cc.local.model_id) if spec else None
        downloaded = model_path.exists() if model_path else False
        api_key = cc.api.api_key
        return {
            "mode": cc.mode,
            "local": {
                "model_id":          cc.local.model_id,
                "model_name":        spec.name if spec else cc.local.model_id,
                "model_downloaded":  downloaded,
                "model_size_gb":     spec.size_gb if spec else 0,
                "model_path":        str(model_path) if downloaded else None,
            },
            "api": {
                "api_key_set":    bool(api_key),
                "api_key_prefix": api_key[:6] if len(api_key) >= 6 else api_key,
                "base_url":       cc.api.base_url,
                "model":          cc.api.model,
            },
        }

    @router.get("")
    def get_settings():
        cfg = _load()
        info = _store_info(store)
        return {
            "iqs_threshold": cfg.get("iqs_threshold", 0.70),
            "metric_weights": cfg.get("metric_weights", DEFAULT_WEIGHTS),
            "corrector": _corrector_state(),
            # Legacy field kept for any old clients. H-1: never echo the raw
            # key back - expose only a masked hint and a boolean.
            "llm_corrector": {
                "provider":     cfg.get("provider", "none"),
                "api_key_set":  bool(cfg.get("api_key", "")),
                "api_key_hint": mask_api_key(cfg.get("api_key", "")),
                "base_url":     cfg.get("base_url", ""),
                "model":        cfg.get("model", ""),
            },
            **info,
        }

    @router.put("")
    def update_settings(body: dict):
        """Patch settings. Handles iqs_threshold, metric_weights, corrector, clear_all_records."""
        cfg = _load()

        if "iqs_threshold" in body:
            cfg["iqs_threshold"] = float(body["iqs_threshold"])

        if "metric_weights" in body:
            cfg["metric_weights"] = body["metric_weights"]

        if "corrector" in body:
            cc = CorrectorConfig.load(default_config_path())
            patch = body["corrector"]
            if "mode" in patch:
                cc.mode = patch["mode"]
            if "local" in patch:
                lp = patch["local"]
                cc.local = LocalConfig(
                    model_id=lp.get("model_id", cc.local.model_id),
                    n_threads=lp.get("n_threads", cc.local.n_threads),
                    n_gpu_layers=lp.get("n_gpu_layers", cc.local.n_gpu_layers),
                    context_window=lp.get("context_window", cc.local.context_window),
                )
            if "api" in patch:
                ap = patch["api"]
                # Write-only key semantics (H-1): a blank/absent api_key means
                # "leave the stored key unchanged" rather than wiping it. This
                # also stops an unrelated edit (e.g. model only) from clearing
                # the key, since the UI never reads the real key back.
                new_key = ap.get("api_key")
                api_key = new_key if new_key else cc.api.api_key
                new_base_url = ap.get("base_url", cc.api.base_url)
                # M-2: reject untrusted/internal endpoints before persisting.
                try:
                    validate_base_url(new_base_url)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                cc.api = APIConfig(
                    api_key=api_key,
                    base_url=new_base_url,
                    model=ap.get("model", cc.api.model),
                    system_prompt=ap.get("system_prompt", cc.api.system_prompt),
                )
            cc.save(default_config_path())

        # Legacy llm_corrector key - still accepted
        if "llm_corrector" in body:
            lc = body["llm_corrector"]
            cfg["provider"]  = lc.get("provider", cfg.get("provider", "none"))
            # Blank/absent key = leave unchanged (H-1 write-only semantics).
            new_key = lc.get("api_key")
            cfg["api_key"]   = new_key if new_key else cfg.get("api_key", "")
            new_base_url     = lc.get("base_url", cfg.get("base_url", ""))
            try:
                validate_base_url(new_base_url)  # M-2
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            cfg["base_url"]  = new_base_url
            cfg["model"]     = lc.get("model",    cfg.get("model", ""))

        if body.get("clear_all_records") and store is not None:
            try:
                store.purge()
            except Exception:
                pass

        _save(cfg)
        return {"status": "ok"}

    @router.post("/test-connection")
    def test_connection():
        cfg = _load()
        provider = cfg.get("provider", "none")
        if provider == "llm":
            from .records import _detect_provider
            provider = _detect_provider(cfg)

        if provider == "none":
            return {"status": "error", "latency_ms": 0, "message": "No provider configured"}

        api_key = cfg.get("api_key", "")
        base_url = cfg.get("base_url") or None
        model = cfg.get("model", "")

        # M-2: never send the key to an unvetted endpoint, even on a test ping.
        try:
            validate_base_url(base_url)
        except ValueError as exc:
            return {"status": "error", "latency_ms": 0, "message": str(exc)}

        start = time.time()
        sample = ""
        status = "ok"
        message = ""

        try:
            if provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
                msg = client.messages.create(
                    model=model or "claude-haiku-4-5-20251001",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "Reply: ok"}],
                )
                sample = msg.content[0].text

            elif provider in ("openai", "groq", "openrouter"):
                import openai
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": "Reply: ok"}],
                    max_tokens=16,
                )
                sample = resp.choices[0].message.content

            elif provider == "ollama":
                import requests
                url = (base_url or "http://localhost:11434") + "/api/generate"
                resp = requests.post(
                    url,
                    json={"model": model or "llama3.2", "prompt": "Reply: ok", "stream": False},
                    timeout=10,
                )
                sample = resp.json().get("response", "")

        except Exception as e:
            status = "error"
            message = str(e)

        latency_ms = int((time.time() - start) * 1000)
        return {"status": status, "latency_ms": latency_ms, "sample_output": sample[:200], "message": message}

    # ─── Legacy llm-judge sub-routes (kept for backwards compat) ──────

    @router.get("/llm-judge")
    def get_llm_judge():
        cfg = _load()
        return {
            "provider":       cfg.get("provider", "none"),
            "model":          cfg.get("model", ""),
            "trigger_mode":   cfg.get("trigger_mode", "manual"),
            "budget_cap_usd": cfg.get("budget_cap_usd"),
            "api_key_env_var": cfg.get("api_key_env_var", ""),
        }

    @router.put("/llm-judge")
    def save_llm_judge(body: dict):
        cfg = _load()
        for k in ("provider", "model", "trigger_mode", "budget_cap_usd", "api_key_env_var"):
            if k in body:
                cfg[k] = body[k]
        _save(cfg)
        return body

    @router.post("/llm-judge/test")
    def test_llm_judge():
        return test_connection()

    return router
