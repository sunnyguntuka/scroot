# Apache-2.0. NO license logic, NO enterprise code — only the seam.
from __future__ import annotations

from importlib.metadata import entry_points

from ._messages import DOCS_URL, SEAM_LABELS


class EnterpriseFeatureError(NotImplementedError):
    """Raised when a scroot Cloud feature is used without scroot-cloud installed."""

    def __init__(self, feature: str) -> None:
        super().__init__(
            f"'{feature}' is a scroot Cloud feature and is not available in the "
            f"open-source package.\n"
            f"  -> Learn more:        {DOCS_URL}\n"
            f"  -> Already licensed?  pip install scroot-cloud"
        )


_REGISTRY: dict[str, object] = {}
_loaded: bool = False


def register_enterprise(name: str, impl: object) -> None:
    _REGISTRY[name] = impl


def _ensure_plugins_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True  # set before loading; a failing plugin must not retrigger discovery
    for ep in entry_points(group="scroot.plugins"):
        try:
            ep.load()()
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("scroot").warning("plugin %s failed: %s", ep.name, exc)


def get_enterprise(name: str, feature_label: str | None = None) -> object:
    _ensure_plugins_loaded()
    impl = _REGISTRY.get(name)
    if impl is None:
        label = feature_label or SEAM_LABELS.get(name, name)
        raise EnterpriseFeatureError(label)
    return impl
