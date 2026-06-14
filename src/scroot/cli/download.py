"""scroot download-model command."""
from __future__ import annotations

from scroot.corrector.models import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    get_model_dir,
    get_model_path,
    is_model_downloaded,
)


def download_model(model_id: str = DEFAULT_MODEL_ID) -> None:
    if model_id not in MODEL_REGISTRY:
        ids = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {ids}")

    spec = MODEL_REGISTRY[model_id]

    if is_model_downloaded(model_id):
        print(f"{spec.name} is already downloaded at {get_model_path(model_id)}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is not installed. "
            "Run: pip install 'scroot[local]'"
        )

    dest = get_model_dir() / model_id
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {spec.name} ({spec.size_gb} GB)...")
    print(f"Source      : {spec.hf_repo}")
    print(f"Destination : {dest}")
    print()

    hf_hub_download(
        repo_id=spec.hf_repo,
        filename=spec.hf_filename,
        local_dir=str(dest),
        resume_download=True,
        token=False,
    )

    print(f"\nOK {spec.name} ready at {dest / spec.hf_filename}")
    print("Run `scroot serve` to start the dashboard.")
