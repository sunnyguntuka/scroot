"""scroot model-info command."""
from __future__ import annotations

from scroot.corrector.models import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    get_model_dir,
    is_model_downloaded,
)


def print_model_info() -> None:
    col = "{:<16} {:<16} {:<8} {:<8} {:<12}"
    print()
    print("  scroot models")
    print()
    print("  " + col.format("Model", "Status", "Size", "RAM", "License"))
    print("  " + "-" * 60)
    for model_id, spec in MODEL_REGISTRY.items():
        default_tag = "  <- default" if model_id == DEFAULT_MODEL_ID else ""
        status = "ready" if is_model_downloaded(model_id) else "not downloaded"
        size = f"{spec.size_gb} GB"
        ram = f"{spec.rec_ram_gb} GB"
        print(f"  {spec.name:<20} {status:<16} {size:<8} {ram:<8} {spec.license}{default_tag}")
    print()
    print(f"  Models stored at: {get_model_dir()}")
    print("  To download: scroot download-model [--model smollm3]")
    print()
