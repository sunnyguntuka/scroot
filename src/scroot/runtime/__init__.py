# Apache-2.0. Local air-gapped runtime (OSS) + managed runtime (Cloud).
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

from .._entitlements import get_enterprise

# In-process cache: (path_str, size, mtime) → sha256_hex.
# Avoids rehashing large weight files on repeated preflight() calls.
_HASH_CACHE: dict[tuple, str] = {}

_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin", "tf_model*.h5")
_MANIFEST_PATH = pathlib.Path(__file__).parent / "model_hashes.json"


def _load_manifest() -> dict[str, str]:
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except Exception:
        return {}


def _find_weight_files(model_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return primary weight file(s) from an HF hub model directory."""
    snapshot_dir: pathlib.Path | None = None
    refs_main = model_dir / "refs" / "main"
    if refs_main.is_file():
        try:
            snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
            candidate = model_dir / "snapshots" / snapshot_hash
            if candidate.is_dir():
                snapshot_dir = candidate
        except OSError:
            pass
    if snapshot_dir is None:
        snapshots = model_dir / "snapshots"
        if snapshots.is_dir():
            dirs = sorted(d for d in snapshots.iterdir() if d.is_dir())
            if dirs:
                snapshot_dir = dirs[-1]
    if snapshot_dir is None:
        return []
    for pattern in _WEIGHT_GLOBS:
        files = sorted(snapshot_dir.glob(pattern))
        if files:
            return files
    return []


def _sha256_file(path: pathlib.Path) -> str:
    """Compute SHA-256 of *path*, returning cached result when file is unchanged."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    _HASH_CACHE[key] = digest
    return digest


def preflight(
    integrity: "Literal['off', 'warn', 'strict']" = "warn",
    expected_hashes: "dict[str, str] | None" = None,
) -> dict:
    """Check whether all required models are locally cached for air-gapped use.

    Args:
        integrity: Weight-file SHA-256 verification mode.
            ``"off"`` — skip entirely.
            ``"warn"`` (default) — surface mismatches in the result without
            flipping ``ready`` (preserves existing pass/fail semantics).
            ``"strict"`` — mismatch or unknown hash sets ``ready=False``.
        expected_hashes: Mapping ``{model_id: sha256_hex}``; overrides /
            extends entries in the bundled ``model_hashes.json`` manifest.
            Useful for custom or side-loaded models.

    Returns:
        Dict with keys:
            ready (bool): True when all required models are in the local cache.
                ``"strict"`` mode also sets this to False on integrity failures.
            missing (list[str]): Model IDs not found in the cache.
            cache_dir (str): Resolved HuggingFace cache directory.
            integrity (dict[str, str]): Per-model integrity status —
                ``"ok"``, ``"mismatch"``, ``"unknown"``, or ``"skipped"``.
                Only present when ``integrity != "off"``.
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

    missing: list[str] = []
    seen: set[str] = set()
    found_dirs: dict[str, pathlib.Path] = {}

    for model_id, slug in required:
        if model_id in seen:
            continue
        model_path = hub / f"models--{slug}"
        if model_path.exists():
            seen.add(model_id)
            found_dirs[model_id] = model_path
            continue
        alt_slug = "models--" + model_id.replace("/", "--")
        alt_path = hub / alt_slug
        if alt_path.exists():
            seen.add(model_id)
            found_dirs[model_id] = alt_path
            continue
        missing.append(model_id)
        seen.add(model_id)

    result: dict = {"ready": not missing, "missing": missing, "cache_dir": str(cache_dir)}

    if integrity == "off":
        return result

    manifest = _load_manifest()
    merged = {**manifest, **(expected_hashes or {})}

    integrity_status: dict[str, str] = {}
    for model_id, model_dir in found_dirs.items():
        expected_hex = merged.get(model_id)
        if expected_hex is None:
            integrity_status[model_id] = "unknown"
            continue
        weight_files = _find_weight_files(model_dir)
        if not weight_files:
            integrity_status[model_id] = "unknown"
            continue
        try:
            computed = _sha256_file(weight_files[0])
        except OSError:
            integrity_status[model_id] = "unknown"
            continue
        integrity_status[model_id] = "ok" if computed == expected_hex else "mismatch"

    result["integrity"] = integrity_status

    if integrity == "strict" and any(
        s in ("mismatch", "unknown") for s in integrity_status.values()
    ):
        result["ready"] = False

    return result


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
