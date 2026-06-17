# Apache-2.0. Local air-gapped runtime (OSS) + managed runtime (Cloud).
from __future__ import annotations

import os
import pathlib

from .._entitlements import get_enterprise


def preflight() -> dict:
    """Check whether all required models are locally cached for air-gapped use.

    Returns:
        Dict with keys:
            ready (bool): True when all required models are in the local cache.
            missing (list[str]): Model IDs not found in the cache.
            cache_dir (str): Resolved HuggingFace cache directory.
    """
    hf_home = os.environ.get(
        "HF_HOME",
        os.environ.get("TRANSFORMERS_CACHE", "~/.cache/huggingface"),
    )
    cache_dir = pathlib.Path(hf_home).expanduser()
    hub = cache_dir / "hub"

    required = [
        ("cross-encoder/nli-deberta-v3-base", "cross-encoder--nli-deberta-v3-base"),
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers--all-MiniLM-L6-v2"),
        # also check without org prefix (some HF versions store it this way)
        ("all-MiniLM-L6-v2", "models--all-MiniLM-L6-v2"),
    ]

    missing = []
    seen: set[str] = set()
    for model_id, slug in required:
        if model_id in seen:
            continue
        # HuggingFace hub layout: hub/models--<org>--<name>/
        model_path = hub / f"models--{slug}"
        if model_path.exists():
            seen.add(model_id)
            continue
        # Check alternate slug (org/name → org--name)
        alt_slug = "models--" + model_id.replace("/", "--")
        if (hub / alt_slug).exists():
            seen.add(model_id)
            continue
        if model_id not in seen:
            missing.append(model_id)
            seen.add(model_id)

    return {"ready": not missing, "missing": missing, "cache_dir": str(cache_dir)}


def run(request: dict) -> dict:
    """Score a response locally with no network calls (air-gapped runtime).

    Fully OSS — wraps Auditor.score() with a JSON-serialisable interface
    suitable for subprocess piping or local socket use.

    Args:
        request: Dict with keys:
            query (str): The user's query.
            response (str): The LLM response.
            context (list[str] | None): Optional grounding context chunks.

    Returns:
        result.to_dict() — the full EntailmentResult as a plain dict.

    Raises:
        KeyError: If ``query`` or ``response`` keys are missing.
    """
    from ..core import Auditor

    auditor = Auditor()
    result = auditor.score(
        query=request["query"],
        response=request["response"],
        context=request.get("context"),
    )
    return result.to_dict()


def managed(*args, **kwargs) -> object:
    """Cloud: hardened/operated managed runtime with autoscaling and SLA gating."""
    return get_enterprise("runtime.managed").start(*args, **kwargs)
