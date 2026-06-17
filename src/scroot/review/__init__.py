# Apache-2.0. Review surfaces: local viewer (OSS) + hosted queue (Cloud).
from __future__ import annotations

from .._entitlements import get_enterprise


def ui(
    port: int = 7432,
    store: str = "./scroot_store.jsonl",
    host: str = "127.0.0.1",
    token: str | None = None,
) -> None:
    """Launch the local scroot Review Console (OSS, shipped in v0.2.0).

    Single-user, ephemeral local viewer. Inspect scores, evidence_map,
    numeric-grounding flags, and filter failures. Corrections feed the
    local feedback loop only. For the hosted multi-reviewer queue see
    ``review.queue()`` (scroot Cloud).

    Args:
        port: Port to listen on. Default 7432.
        store: Path to the JSONL feedback store.
        host: Host to bind. Default 127.0.0.1 (loopback-safe).
        token: Optional shared token for network binds.
    """
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "Install dashboard dependencies: pip install 'scroot[dashboard]'"
        ) from e

    from ..dashboard.server import create_app

    fa_app = create_app(store_path=store, hosted=False, host=host, auth_token=token)
    uvicorn.run(fa_app, host=host, port=port, log_level="info")


def queue(*args, **kwargs) -> object:
    """Cloud: hosted multi-reviewer queue with assignment, claim/lock, and sign-off."""
    return get_enterprise("review.queue", "Hosted review queue").open(*args, **kwargs)
