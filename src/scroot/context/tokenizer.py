"""Token counting for context budgeting.

Uses tiktoken when installed (accurate for OpenAI-family tokenisers);
falls back to a character-based estimate (~4 chars per token) otherwise.
The fallback intentionally over-estimates slightly so the max_tokens
budget errs on the safe side.
"""

from __future__ import annotations

_encoder = None
_tiktoken_checked = False


def _get_encoder():
    global _encoder, _tiktoken_checked
    if not _tiktoken_checked:
        _tiktoken_checked = True
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text.

    Args:
        text: Input string.

    Returns:
        Token count via tiktoken if available, otherwise
        ``max(1, ceil(len(text) / 4))`` for non-empty text.
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, -(-len(text) // 4))
