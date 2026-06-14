"""Scroot Dashboard - FastAPI application factory."""
from __future__ import annotations

import warnings
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from scroot.feedback.store import FeedbackStore
from .security import (
    extract_request_token,
    is_loopback_host,
    resolve_dashboard_token,
    token_matches,
)
from .routers.queue import queue_router
from .routers.records import records_router
from .routers.analytics import analytics_router
from .routers.export import export_router
from .routers.settings import settings_router
from .routers.pipeline import pipeline_router
from .routers.corrector import corrector_router
from .routers.guardrails import guardrails_router

# Resolved at import time - ui/dist is built by `npm run build`
UI_DIST_PATH = str(Path(__file__).parent.parent / "ui" / "dist")

# Endpoints reachable without a token even when auth is enabled.
_UNAUTHENTICATED_PATHS = frozenset({"/api/health"})


def create_app(
    store_path: str,
    hosted: bool = False,
    host: str = "127.0.0.1",
    auth_token: str | None = None,
) -> FastAPI:
    """Create the Scroot dashboard FastAPI application.

    Args:
        store_path: Path to the JSONL FeedbackStore file.
        hosted: Reserved for Scroot Enterprise hosted mode.
        host: The interface the app will be served on. Used only to decide
            whether to warn about an unauthenticated non-loopback bind (H-2).
        auth_token: Optional shared token. When set (or
            ``SCROOT_DASHBOARD_TOKEN`` is in the environment), every
            ``/api/*`` route except the health check requires the token via an
            ``Authorization: Bearer <token>`` or ``X-Scroot-Token`` header.

    Raises:
        NotImplementedError: If hosted=True (enterprise-only feature).
    """
    if hosted:
        raise NotImplementedError(
            "Hosted mode is available in Scroot Cloud. "
            "Visit https://scroot.dev/cloud for enterprise pricing."
        )

    store = FeedbackStore(store_path)
    token = resolve_dashboard_token(auth_token)

    # H-2: the dashboard has no per-user auth. A loopback bind is single-user
    # safe; binding to a routable interface exposes the full correction store
    # and the corrector API key to the network. Warn unless a token is set.
    if not is_loopback_host(host) and token is None:
        warnings.warn(
            f"Scroot dashboard is binding to a non-loopback host ({host!r}) "
            f"with no authentication. The correction store and stored LLM API "
            f"key would be reachable by anyone on the network. Set "
            f"SCROOT_DASHBOARD_TOKEN (or pass --token) and/or run behind an "
            f"authenticating reverse proxy.",
            stacklevel=2,
        )

    app = FastAPI(
        title="Scroot Review Console",
        description="Local feedback loop review dashboard",
        version="0.2.0",
    )

    # H-2: optional shared-token gate for network-exposed deployments.
    if token is not None:
        @app.middleware("http")
        async def _require_token(request, call_next):
            path = request.url.path
            if path.startswith("/api/") and path not in _UNAUTHENTICATED_PATHS:
                provided = extract_request_token(request.headers)
                if not token_matches(provided, token):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing or invalid dashboard token."},
                    )
            return await call_next(request)

    # CORS for Vite dev server (dev only - not needed in production)
    try:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except ImportError:
        pass

    # API routers
    app.include_router(queue_router(store),        prefix="/api/queue",    tags=["queue"])
    app.include_router(records_router(store),      prefix="/api/records",  tags=["records"])
    app.include_router(analytics_router(store),    prefix="/api/analytics", tags=["analytics"])
    app.include_router(export_router(store),       prefix="/api/export",   tags=["export"])
    app.include_router(settings_router(store),     prefix="/api/settings", tags=["settings"])
    app.include_router(pipeline_router(store),     prefix="/api/pipeline", tags=["pipeline"])
    app.include_router(corrector_router(),         prefix="/api/corrector", tags=["corrector"])
    app.include_router(guardrails_router(store),   prefix="/api/guardrails", tags=["guardrails"])

    # Health check
    @app.get("/api/health")
    def health():
        pending = sum(
            1 for r in store.get_all()
            if getattr(r, "status", "pending") == "pending"
        )
        iqs_vals = [
            r.scores.get("iqs", 0)
            for r in store.get_all()
            if isinstance(r.scores, dict)
        ]
        avg_iqs = round(sum(iqs_vals) / len(iqs_vals), 3) if iqs_vals else None
        return {
            "status": "ok",
            "version": "0.2.0",
            "pending_count": pending,
            "avg_iqs_today": avg_iqs,
        }

    # Serve built React SPA - must be registered last.
    dist = Path(UI_DIST_PATH)
    if dist.exists():
        from fastapi.responses import FileResponse

        index_file = dist / "index.html"
        # Hashed build assets (JS/CSS) live under /assets.
        assets_dir = dist / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="assets",
            )

        # SPA history-API fallback: the dashboard uses BrowserRouter, so deep
        # links and refreshes hit the server with a client-side path (e.g.
        # /queue, /analytics). Serve a real file when one exists (favicon
        # etc.), otherwise return index.html so the SPA can route. /api/* is
        # never caught here (those routes are registered above).
        @app.get("/{full_path:path}")
        def serve_spa(full_path: str):
            # Don't shadow the API: unknown /api paths get a real 404, not the SPA.
            if full_path.startswith("api/"):
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Not Found")
            candidate = (dist / full_path).resolve()
            if (
                full_path
                and dist.resolve() in candidate.parents
                and candidate.is_file()
            ):
                return FileResponse(str(candidate))
            return FileResponse(str(index_file))
    else:
        @app.get("/")
        def ui_not_built():
            return {
                "error": "UI not built",
                "hint": "cd src/scroot/ui && npm install && npm run build",
            }

    return app
