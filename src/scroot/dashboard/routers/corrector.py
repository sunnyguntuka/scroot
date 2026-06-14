"""Corrector router - /api/corrector endpoints."""
from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from scroot.config.corrector import CorrectorConfig, default_config_path
from scroot.corrector.models import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    get_model_path,
    is_model_downloaded,
)

_downloads: dict[str, dict[str, Any]] = {}


def _model_entry(model_id: str, spec) -> dict:
    path = get_model_path(model_id)
    downloaded = path.exists()
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "size_gb": spec.size_gb,
        "min_ram_gb": spec.min_ram_gb,
        "rec_ram_gb": spec.rec_ram_gb,
        "context_window": spec.context_window,
        "license": spec.license,
        "is_default": model_id == DEFAULT_MODEL_ID,
        "downloaded": downloaded,
        "path": str(path) if downloaded else None,
    }


def _do_download(model_id: str) -> None:
    """Background download task."""
    spec = MODEL_REGISTRY[model_id]
    state = _downloads[model_id]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        state["status"] = "failed"
        state["error"] = "huggingface-hub not installed; run pip install 'scroot[local]'"
        return

    dest = get_model_path(model_id).parent
    dest.mkdir(parents=True, exist_ok=True)
    state["status"] = "downloading"

    try:
        state["total_bytes"] = int(spec.size_gb * 1_073_741_824)

        hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.hf_filename,
            local_dir=str(dest),
            resume_download=True,
            token=False,
        )
        state["status"] = "ready"
        state["progress_pct"] = 100
        state["eta_seconds"] = 0
    except Exception as e:
        state["status"] = "failed"
        state["error"] = str(e)


def corrector_router() -> APIRouter:
    router = APIRouter()

    @router.get("/runtime")
    def runtime_status():
        try:
            import llama_cpp  # noqa: F401
            llama_cpp_installed = True
        except ImportError:
            llama_cpp_installed = False
        try:
            import huggingface_hub  # noqa: F401
            hf_hub_installed = True
        except ImportError:
            hf_hub_installed = False
        return {
            "llama_cpp_installed": llama_cpp_installed,
            "hf_hub_installed": hf_hub_installed,
            "ready": llama_cpp_installed and hf_hub_installed,
        }

    @router.get("/models")
    def list_models():
        return {
            "models": [
                _model_entry(mid, spec)
                for mid, spec in MODEL_REGISTRY.items()
            ]
        }

    @router.post("/models/{model_id}/download")
    def start_download(model_id: str):
        if model_id not in MODEL_REGISTRY:
            raise HTTPException(404, f"Unknown model: {model_id}")
        if is_model_downloaded(model_id):
            return {"model_id": model_id, "status": "ready"}
        existing = _downloads.get(model_id, {})
        if existing.get("status") == "downloading":
            return {"model_id": model_id, "status": "downloading"}

        _downloads[model_id] = {
            "status": "pending",
            "progress_bytes": 0,
            "total_bytes": 0,
            "progress_pct": 0,
            "eta_seconds": None,
            "error": None,
            "_started": time.time(),
        }
        t = threading.Thread(target=_do_download, args=(model_id,), daemon=True)
        t.start()
        return {"model_id": model_id, "status": "downloading"}

    @router.get("/models/{model_id}/download-status")
    def download_status(model_id: str):
        if model_id not in MODEL_REGISTRY:
            raise HTTPException(404, f"Unknown model: {model_id}")
        if is_model_downloaded(model_id) and model_id not in _downloads:
            return {
                "model_id": model_id, "status": "ready",
                "progress_bytes": 0, "total_bytes": 0,
                "progress_pct": 100, "eta_seconds": 0, "error": None,
            }
        state = _downloads.get(model_id, {
            "status": "pending", "progress_bytes": 0, "total_bytes": 0,
            "progress_pct": 0, "eta_seconds": None, "error": None,
        })
        return {
            "model_id": model_id,
            "status": state["status"],
            "progress_bytes": state["progress_bytes"],
            "total_bytes": state["total_bytes"],
            "progress_pct": state["progress_pct"],
            "eta_seconds": state["eta_seconds"],
            "error": state["error"],
        }

    @router.delete("/models/{model_id}")
    def delete_model(model_id: str):
        if model_id not in MODEL_REGISTRY:
            raise HTTPException(404, f"Unknown model: {model_id}")
        path = get_model_path(model_id)
        if not path.exists():
            raise HTTPException(404, f"Model {model_id} is not downloaded")

        # Unload if active
        try:
            from scroot.corrector import _active_corrector
            from scroot.corrector.local import LocalLLMCorrector
            if isinstance(_active_corrector, LocalLLMCorrector):
                _active_corrector.unload()
        except Exception:
            pass

        import shutil
        freed = path.stat().st_size
        path.unlink()
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            shutil.rmtree(parent, ignore_errors=True)

        freed_gb = round(freed / 1_073_741_824, 2)
        return {"model_id": model_id, "deleted": True, "freed_gb": freed_gb}

    @router.post("/test")
    def test_corrector():
        cfg = CorrectorConfig.load(default_config_path())
        if cfg.mode == "disabled":
            return {
                "mode": "disabled", "model": None, "latency_ms": 0,
                "sample_output": None, "tokens_generated": 0,
                "tok_per_sec": None, "error": "Corrector is disabled",
            }

        from scroot.corrector import get_corrector
        corrector = get_corrector(cfg)
        if not corrector.is_available:
            return {
                "mode": cfg.mode, "model": None, "latency_ms": 0,
                "sample_output": None, "tokens_generated": 0,
                "tok_per_sec": None,
                "error": "Corrector is not available. Check model download or API key.",
            }

        test_query = "What is the capital of France?"
        test_response = "The capital of France is Berlin."
        test_context = "France is a country in Western Europe. Its capital city is Paris."

        start = time.time()
        error = None
        sample = None
        model_name = None
        tok_per_sec = None

        try:
            sample = corrector.draft_correction(test_query, test_response, test_context)
            if cfg.mode == "local":
                spec = MODEL_REGISTRY[cfg.local.model_id]
                model_name = spec.name
                tok_per_sec = getattr(corrector, "tok_per_sec", lambda: None)()
            else:
                model_name = cfg.api.model
        except Exception as e:
            error = str(e)

        latency_ms = int((time.time() - start) * 1000)
        tokens = len(sample.split()) if sample else 0

        return {
            "mode": cfg.mode,
            "model": model_name,
            "latency_ms": latency_ms,
            "sample_output": sample[:400] if sample else None,
            "tokens_generated": tokens,
            "tok_per_sec": tok_per_sec,
            "error": error,
        }

    return router
