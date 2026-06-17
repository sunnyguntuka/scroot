# Apache-2.0. Public PII API — scrub (OSS) + policy (Cloud).
from __future__ import annotations

from .context.pii import ScrubResult, scrub as _scrub_impl
from ._entitlements import get_enterprise


def scrub(
    text: str,
    *,
    allowlist: set[str] | None = None,
    preserve_for_grounding: bool = False,
) -> ScrubResult:
    """Replace detected PII with typed placeholders.

    Fully OSS — regex-based, local, no external API.

    Args:
        text: Raw text that may contain PII.
        allowlist: Set of entity-type names to leave unmasked
            (e.g. ``{"EMAIL"}`` to keep email addresses). Entity type
            names match the keys in ``ScrubResult.summary``
            (EMAIL, PHONE, SSN, CARD, IP, DOB, ADDRESS, PERSON, SECRET).
        preserve_for_grounding: When True, replaces each entity with a
            per-document sequential placeholder ``[TYPE:N]`` (e.g.
            ``[EMAIL:1]``, ``[EMAIL:2]``). The counter is deterministic
            within a single call, so the same email in both the context
            and the response receives the same placeholder, allowing
            numeric-grounding NLI to match entity references without
            exposing the original value.

    Returns:
        ScrubResult with the scrubbed text and a count-only summary.
    """
    return _scrub_impl(
        text,
        allowlist=allowlist,
        preserve_for_grounding=preserve_for_grounding,
    )


def policy(*args, **kwargs) -> object:
    """Cloud: regulatory PII policy management, GDPR/HIPAA presets, DLP hooks."""
    return get_enterprise("pii.policy").apply(*args, **kwargs)
