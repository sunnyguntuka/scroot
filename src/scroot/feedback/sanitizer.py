"""Prompt-injection sanitizer for feedback record fields.

Strips role-boundary markers, common injection override phrases,
and normalizes whitespace before any field is interpolated into
an LLM system prompt.
"""

import re

# Lines that begin with LLM role markers are dropped entirely.
_ROLE_MARKER_RE = re.compile(
    r"^\s*(?:#{1,3}\s*)?(?:system|assistant|user|human|ai|instruction|prompt|role"
    r"|<\|im_start\|>|<\|im_end\|>|<s>|</s>|\[INST\]|\[/INST\])",
    re.IGNORECASE,
)

# Inline phrases that attempt to override or hijack instructions.
_INJECTION_RE = re.compile(
    r"\b(?:"
    r"ignore\s+(?:all|previous|prior|above|the\s+above)"
    r"|forget\s+(?:all|everything|previous|prior|the\s+above)"
    r"|disregard\s+(?:all|previous|prior|the\s+above|your|instructions)"
    r"|output\s+the\s+(?:system\s+prompt|prompt|instructions|context)"
    r"|reveal\s+the\s+(?:system\s+prompt|prompt|instructions|context)"
    r"|you\s+are\s+now\s+(?:in|an?|the)"
    r"|act\s+as\s+(?:a|an)\s+\w"
    r"|new\s+(?:instructions|directive|persona|role|task)"
    r"|bypass\s+(?:all|the|your)"
    r"|jailbreak"
    r")\b",
    re.IGNORECASE,
)


def sanitize_for_prompt(text: str, max_length: int = 500) -> str:
    """Sanitize a CorrectionRecord field for safe system-prompt interpolation.

    Steps applied in order:
    1. Drop any line that starts with an LLM role-boundary marker.
    2. Replace injection-override phrases with [FILTERED].
    3. Collapse all whitespace (newlines → spaces, runs → single space).
    4. Truncate to max_length *after* sanitization.

    Args:
        text: Raw field value from a CorrectionRecord.
        max_length: Maximum character length of the returned string.

    Returns:
        Sanitized, truncated string safe for prompt interpolation.
    """
    lines = text.splitlines()
    clean_lines = [line for line in lines if not _ROLE_MARKER_RE.match(line)]
    joined = " ".join(clean_lines)
    sanitized = _INJECTION_RE.sub("[FILTERED]", joined)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:max_length]
