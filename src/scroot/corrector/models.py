"""Model registry - GGUF model specs and local storage helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSpec:
    id: str
    name: str
    hf_repo: str
    hf_filename: str
    size_gb: float
    min_ram_gb: int
    rec_ram_gb: int
    context_window: int
    license: str
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "phi4-mini": ModelSpec(
        id="phi4-mini",
        name="Qwen2.5-3B-Instruct",
        hf_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        hf_filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        size_gb=2.0,
        min_ram_gb=4,
        rec_ram_gb=6,
        context_window=32_768,
        license="Apache 2.0",
        description="Alibaba's efficient 3B instruction model. Strong "
                    "reasoning and instruction following. Default choice.",
    ),
    "smollm3": ModelSpec(
        id="smollm3",
        name="Qwen2.5-1.5B-Instruct",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        hf_filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_gb=1.0,
        min_ram_gb=2,
        rec_ram_gb=4,
        context_window=32_768,
        license="Apache 2.0",
        description="Compact 1.5B model. Fastest on CPU, lower RAM "
                    "requirement. Good for resource-constrained machines.",
    ),
}

DEFAULT_MODEL_ID = "phi4-mini"


def get_model_dir() -> Path:
    """Respects SCROOT_MODELS_DIR env override for custom storage."""
    custom = os.environ.get("SCROOT_MODELS_DIR")
    if custom:
        return Path(custom)
    return Path.home() / ".scroot" / "models"


def get_model_path(model_id: str) -> Path:
    spec = MODEL_REGISTRY[model_id]
    return get_model_dir() / model_id / spec.hf_filename


def is_model_downloaded(model_id: str) -> bool:
    return get_model_path(model_id).exists()
