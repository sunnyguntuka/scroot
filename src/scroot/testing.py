# Apache-2.0. Test helpers — not part of the public API; not shipped to end users.
from __future__ import annotations


def reset_registry() -> None:
    """Reset the enterprise registry and plugin-loaded flag.

    Call at the start of each test that needs to simulate an OSS-only environment
    or register a mock enterprise implementation without interference from installed
    plugins.
    """
    from . import _entitlements

    _entitlements._REGISTRY.clear()
    _entitlements._loaded = False

    from .metrics._registry import clear_custom_metrics

    clear_custom_metrics()
