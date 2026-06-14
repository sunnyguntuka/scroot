"""Shared dashboard security helpers.

Covers three hardening controls for the local Review Console:

* **H-1** - ``mask_api_key()`` so stored provider keys are never echoed back
  in plaintext over the API.
* **M-2** - ``validate_base_url()`` so the server cannot be pointed at an
  attacker-controlled host (key exfiltration) or used as an SSRF pivot to
  internal / cloud-metadata endpoints.
* **H-2** - ``require_token`` middleware factory + ``is_loopback_host()`` so a
  network-exposed dashboard can require a shared token and a non-loopback bind
  warns the operator.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# H-1: API key masking
# ---------------------------------------------------------------------------

def mask_api_key(key: str | None) -> str:
    """Return a non-reversible hint for an API key (never the full value).

    ``"sk-abcdefgh...wxyz"`` -> ``"sk-a…wxyz"``. Empty/short keys collapse to a
    placeholder so the real value never leaves the process.
    """
    if not key:
        return ""
    if len(key) <= 8:
        return "…"
    return f"{key[:4]}…{key[-4:]}"


# ---------------------------------------------------------------------------
# M-2: outbound base_url allowlist
# ---------------------------------------------------------------------------

#: Known hosted LLM provider hosts that may receive an API key.
ALLOWED_LLM_HOSTS: frozenset[str] = frozenset({
    "api.openai.com",
    "api.anthropic.com",
    "api.groq.com",
    "openrouter.ai",
    "api.openrouter.ai",
    "generativelanguage.googleapis.com",  # Google Gemini
})

#: Loopback hosts permitted for self-hosted endpoints (e.g. local Ollama).
_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

#: Escape hatch for operators who deliberately use a custom gateway.
_OVERRIDE_ENV = "SCROOT_ALLOW_ANY_BASE_URL"


def validate_base_url(base_url: str | None, *, allow_local: bool = True) -> None:
    """Reject base URLs that aren't a known provider or an allowed local host.

    Args:
        base_url: The configured endpoint. Empty/None means "use the provider
            SDK's default endpoint" and is always allowed.
        allow_local: Permit loopback hosts (needed for local Ollama). Set False
            for hosted providers that should never be local.

    Raises:
        ValueError: If the host is neither an allowlisted provider nor an
            allowed loopback host, and the override env var is not set.
    """
    if not base_url:
        return
    if os.environ.get(_OVERRIDE_ENV) == "1":
        return

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"base_url must use http(s), got {base_url!r}"
        )

    host = (parsed.hostname or "").lower()
    if host in ALLOWED_LLM_HOSTS:
        return
    if allow_local and host in _LOCAL_HOSTS:
        return

    raise ValueError(
        f"base_url host {host!r} is not an allowed LLM provider endpoint. "
        f"This blocks pointing the server at an untrusted host (key theft) or "
        f"an internal/metadata address (SSRF). Allowed providers: "
        f"{', '.join(sorted(ALLOWED_LLM_HOSTS))}. To override for a trusted "
        f"custom gateway, set {_OVERRIDE_ENV}=1."
    )


# ---------------------------------------------------------------------------
# H-2: bind-host inspection + token auth
# ---------------------------------------------------------------------------

def is_loopback_host(host: str) -> bool:
    """True if ``host`` is a loopback / localhost address."""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_dashboard_token(explicit: str | None = None) -> str | None:
    """Return the configured dashboard token, if any.

    Precedence: explicit argument, then ``SCROOT_DASHBOARD_TOKEN`` env var.
    Returns None when no token is configured (auth disabled).
    """
    token = explicit or os.environ.get("SCROOT_DASHBOARD_TOKEN") or ""
    return token or None


def token_matches(provided: str | None, expected: str) -> bool:
    """Constant-time comparison of a presented token against the expected one."""
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


def extract_request_token(headers) -> str | None:
    """Pull a token from ``Authorization: Bearer`` or ``X-Scroot-Token``."""
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-scroot-token") or headers.get("X-Scroot-Token")
