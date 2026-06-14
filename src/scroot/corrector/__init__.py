"""Corrector provider factory."""
from __future__ import annotations

from scroot.corrector.base import BaseCorrector
from scroot.corrector.disabled import NullCorrector

_active_corrector: BaseCorrector | None = None


def get_corrector(config) -> BaseCorrector:
    """
    Return the active corrector for the current config.
    Unloads LocalLLMCorrector from RAM when switching away from local mode.
    """
    global _active_corrector

    mode = config.mode

    if mode == "disabled":
        _active_corrector = NullCorrector()

    elif mode == "local":
        from scroot.corrector.local import LocalLLMCorrector
        if (
            isinstance(_active_corrector, LocalLLMCorrector)
            and _active_corrector.model_spec.id != config.local.model_id
        ):
            _active_corrector.unload()
        _active_corrector = LocalLLMCorrector(config.local)

    elif mode == "api":
        from scroot.corrector.api import APICorrector
        _active_corrector = APICorrector(config.api)

    else:
        _active_corrector = NullCorrector()

    return _active_corrector
