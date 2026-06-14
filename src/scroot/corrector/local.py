"""LocalLLMCorrector - llama-cpp-python inference, thread-safe, lazy-loaded."""
from __future__ import annotations

import threading

from scroot.corrector.base import BaseCorrector
from scroot.corrector.models import MODEL_REGISTRY, get_model_path, is_model_downloaded


class LocalLLMCorrector(BaseCorrector):
    """
    Wraps llama-cpp-python for in-process CPU (or GPU) inference.
    Thread-safe via lock. Lazy-loaded: model is not loaded until first call.
    """

    _lock = threading.Lock()

    def __init__(self, config) -> None:
        self._config = config
        self._llm = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Run: pip install 'scroot[local]'"
            )

        model_id = self._config.model_id
        if not is_model_downloaded(model_id):
            spec = MODEL_REGISTRY[model_id]
            raise RuntimeError(
                f"Model '{spec.name}' is not downloaded. "
                f"Run: scroot download-model --model {model_id}"
            )

        model_path = get_model_path(model_id)
        import os
        n_threads = self._config.n_threads
        if n_threads == -1:
            n_threads = os.cpu_count() or 4

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self._config.context_window,
            n_threads=n_threads,
            n_gpu_layers=self._config.n_gpu_layers,
            verbose=False,
        )

    @property
    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return is_model_downloaded(self._config.model_id)
        except ImportError:
            return False

    def draft_correction(
        self,
        query: str,
        response: str,
        context: str | None,
    ) -> str:
        with self._lock:
            self._ensure_loaded()
            result = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a correction assistant. "
                            "Rewrite the LLM response to be more accurate, "
                            "complete, and grounded in the provided context. "
                            "Return only the corrected response text. "
                            "Do not explain your changes. Do not add preamble."
                        ),
                    },
                    {"role": "user", "content": self._build_prompt(query, response, context)},
                ],
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            return result["choices"][0]["message"]["content"].strip()

    def _build_prompt(self, query: str, response: str, context: str | None) -> str:
        parts = [f"Query:\n{query}", f"\nOriginal response:\n{response}"]
        if context:
            parts.append(f"\nContext / grounding documents:\n{context}")
        parts.append("\nRewrite the response to be more accurate and complete.")
        return "\n".join(parts)

    def unload(self) -> None:
        """Free the model from RAM. Call before deleting the GGUF file."""
        with self._lock:
            self._llm = None

    @property
    def model_spec(self):
        return MODEL_REGISTRY[self._config.model_id]

    def tok_per_sec(self) -> float | None:
        """Rough throughput estimate based on model family."""
        model_id = self._config.model_id
        return {"phi4-mini": 16.0, "smollm3": 22.0}.get(model_id)
