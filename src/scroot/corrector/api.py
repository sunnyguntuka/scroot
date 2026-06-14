"""APICorrector - OpenAI-compatible endpoint, provider auto-detected from key.

Design rationale: why ``api_key`` alone is not enough
-----------------------------------------------------

A common assumption is that an API key fully identifies an LLM connection, so
``model`` and ``base_url`` should be unnecessary. They are not. An LLM request is::

    POST {base_url}/chat/completions
    Headers: {auth_header}: {api_key}
    Body:    {"model": "<name>", "messages": [...]}

The key only fills the **auth header** - it proves *who you are*. It does not
carry the two other things every request needs:

1. **Where to send it (``base_url``).** Each provider has a different endpoint
   and even a different auth-header name (Anthropic uses ``x-api-key``; OpenAI
   uses ``Authorization: Bearer``). This *is* derivable from the key, because the
   key prefix is provider-specific - that is exactly what ``detect_provider``
   does (``sk-ant-`` -> Anthropic, ``AIza`` -> Gemini, ``sk-`` -> OpenAI, else
   OpenRouter). So ``base_url`` can stay optional/advanced: leave it blank for the
   four known providers, set it only for Groq / OpenRouter / a custom gateway.

2. **Which model to run (``model``).** This is a *mandatory* field in the
   request body and it is **not derivable from anything**. The key is tied to an
   *account*, not a model: the same Anthropic key calls Opus, Sonnet, and Haiku.
   There is no way to infer "the user wants Haiku" from the key, the endpoint, or
   the header. The specific model is a **decision** (a cost/quality trade-off),
   not data - so it cannot be auto-detected the way ``base_url`` can. It can only
   be (a) defaulted to an opinionated pick, or (b) chosen by the user.

Consequence for the architecture / UI:

* ``base_url`` - safe to hide behind "Advanced"; auto-detected from the key.
* ``model`` - must remain a real (optional) field. Today it defaults to
  ``gpt-4o-mini`` (see ``draft_correction``), which is only correct for OpenAI;
  a non-OpenAI key with a blank model will send ``gpt-4o-mini`` to the wrong
  provider and fail. The intended improvement is a **per-provider default map**
  (OpenAI -> ``gpt-4o-mini``, Anthropic -> ``claude-haiku-4-5``, Gemini ->
  ``gemini-2.0-flash``), deliberately the cheap/fast tier since this is response
  correction, not frontier reasoning - so "paste key, leave model blank" works
  for every provider while power users can still override.

See also: ``validate_base_url`` in ``scroot.dashboard.security`` (M-2), which
restricts ``base_url`` to allowlisted provider hosts to prevent key exfiltration
and SSRF.
"""
from __future__ import annotations

from scroot.corrector.base import BaseCorrector

_KEY_PREFIX_MAP = {
    "sk-ant-": {
        "base_url": "https://api.anthropic.com/v1",
        "auth_header": "x-api-key",
        "provider_name": "Anthropic",
    },
    "AIza": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "auth_header": "Authorization",
        "provider_name": "Google Gemini",
    },
}
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENAI_BASE = "https://api.openai.com/v1"


def detect_provider(api_key: str, base_url_override: str = "") -> tuple[str, str, str]:
    """Returns (base_url, auth_header, provider_name)."""
    if base_url_override:
        name = "Custom"
        if "groq" in base_url_override:
            name = "Groq"
        elif "openrouter" in base_url_override:
            name = "OpenRouter"
        elif "anthropic" in base_url_override:
            name = "Anthropic"
        return base_url_override, "Authorization", name
    for prefix, cfg in _KEY_PREFIX_MAP.items():
        if api_key.startswith(prefix):
            return cfg["base_url"], cfg["auth_header"], cfg["provider_name"]
    if api_key.startswith("sk-"):
        return _OPENAI_BASE, "Authorization", "OpenAI"
    return _OPENROUTER_BASE, "Authorization", "OpenRouter"


class APICorrector(BaseCorrector):
    def __init__(self, config) -> None:
        self._config = config

    @property
    def is_available(self) -> bool:
        return bool(self._config.api_key)

    def draft_correction(
        self,
        query: str,
        response: str,
        context: str | None,
    ) -> str:
        try:
            import httpx
        except ImportError:
            raise RuntimeError(
                "httpx is not installed. Run: pip install 'scroot[api]'"
            )

        base_url, auth_header, _ = detect_provider(
            self._config.api_key, self._config.base_url
        )
        # M-2: refuse to send the API key to an unvetted/internal endpoint.
        from scroot.dashboard.security import validate_base_url
        validate_base_url(base_url)
        headers = {
            "Content-Type": "application/json",
            auth_header: (
                self._config.api_key
                if auth_header == "x-api-key"
                else f"Bearer {self._config.api_key}"
            ),
        }
        payload = {
            "model": self._config.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": self._build_prompt(query, response, context)},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _build_prompt(self, query: str, response: str, context: str | None) -> str:
        parts = [f"Query:\n{query}", f"\nOriginal response:\n{response}"]
        if context:
            parts.append(f"\nContext:\n{context}")
        parts.append("\nRewrite the response to be more accurate and complete.")
        return "\n".join(parts)
