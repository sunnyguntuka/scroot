"""CorrectorConfig - persisted to ~/.scroot/config.json."""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CorrectorMode = Literal["disabled", "local", "api"]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a correction assistant. Rewrite the response to be more "
    "accurate, complete, and grounded in the provided context. "
    "Return only the corrected response text, nothing else."
)


@dataclass
class LocalConfig:
    model_id: str = "phi4-mini"
    n_threads: int = -1
    n_gpu_layers: int = 0
    context_window: int = 4096


@dataclass
class APIConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT


@dataclass
class CorrectorConfig:
    mode: CorrectorMode = "disabled"
    local: LocalConfig = field(default_factory=LocalConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def save(self, path: Path) -> None:
        # M-1: this file holds the provider API key. Create the parent dir as
        # owner-only and write the file with 0600 so other local accounts
        # cannot read the key.
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        payload = json.dumps(
            {
                "mode": self.mode,
                "local": vars(self.local),
                "api": vars(self.api),
            },
            indent=2,
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        try:
            os.chmod(path, 0o600)  # tighten if the file pre-existed
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path) -> "CorrectorConfig":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                mode=data.get("mode", "disabled"),
                local=LocalConfig(**data.get("local", {})),
                api=APIConfig(**data.get("api", {})),
            )
        except (OSError, ValueError, TypeError) as exc:
            # A corrupt/unreadable config silently resetting to defaults would
            # wipe the stored key/settings on the next save(). Surface it so the
            # operator can recover the file instead of losing it unknowingly.
            warnings.warn(
                f"Could not read corrector config at {path} ({exc}); "
                f"using defaults. The file will be overwritten on the next save.",
                stacklevel=2,
            )
            return cls()


def default_config_path() -> Path:
    return Path.home() / ".scroot" / "config.json"
