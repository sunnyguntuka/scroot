"""PII detection and scrubbing for context content.

Regex-based, fully local - no external API call, consistent with
scroot's zero-external-dependency principle. Detected entities are
replaced with typed placeholders (e.g. ``[EMAIL]``); the scrub summary
records counts by entity type only, never the original values.

Detection is best-effort: regex catches structured PII (emails, phones,
SSNs, cards, IPs, secrets, dates, street addresses) reliably, and person
names via honorifics and a common-first-name heuristic. For regulated
workloads, layer a dedicated NER scrubber in front and pass pre-scrubbed
text in with ``pii_scrub=False``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ScrubResult:
    """Result of scrubbing one piece of text.

    Attributes:
        scrubbed_text: Text with PII replaced by typed placeholders.
        summary: Counts by entity type plus ``total_entities_scrubbed``.
            Never contains original values.
        was_scrubbed: True if at least one entity was replaced.
    """
    scrubbed_text: str
    summary: dict[str, int]
    was_scrubbed: bool


# Common first names used for best-effort [PERSON] detection when no
# honorific is present. Matches "<FirstName> <Capitalized Surname>".
_COMMON_FIRST_NAMES = (
    "James|John|Robert|Michael|William|David|Richard|Joseph|Thomas|Charles|"
    "Christopher|Daniel|Matthew|Anthony|Mark|Donald|Steven|Paul|Andrew|Joshua|"
    "Kenneth|Kevin|Brian|George|Timothy|Ronald|Edward|Jason|Jeffrey|Ryan|"
    "Mary|Patricia|Jennifer|Linda|Elizabeth|Barbara|Susan|Jessica|Sarah|Karen|"
    "Lisa|Nancy|Betty|Margaret|Sandra|Ashley|Kimberly|Emily|Donna|Michelle|"
    "Carol|Amanda|Dorothy|Melissa|Deborah|Stephanie|Rebecca|Sharon|Laura|"
    "Jane|Emma|Olivia|Sophia|Alice|Anna|Maria|Rachel|Hannah|Grace"
)

# Ordered by priority - earlier patterns run first so that, e.g., an API
# key is redacted as [SECRET] before the generic patterns see it.
_PATTERNS: dict[str, re.Pattern] = {
    "SECRET": re.compile(
        r'\b(?:sk-ant-[a-zA-Z0-9-]{20,}|sk-[a-zA-Z0-9]{20,}|'
        r'AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|[a-f0-9]{32,})\b'
    ),
    "EMAIL": re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b'),
    "CARD": re.compile(r'\b(?:\d[ -]?){13,16}\b'),
    "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "IP": re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b'),
    "PHONE": re.compile(
        r'(?:\+?\d{1,2}[-.\s])?\(?\d{3}\)?[-.\s]\d{3,4}[-.\s]?\d{0,4}\b'
        r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    ),
    "DOB": re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
        r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
        r'Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}\b'
        r'|\b\d{1,2}/\d{1,2}/\d{4}\b'
    ),
    "ADDRESS": re.compile(
        r'\b\d{1,5}\s+(?:[A-Z][a-zA-Z]+\s+){1,3}'
        r'(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|'
        r'Dr|Drive|Ct|Court|Way|Pl|Place)\b\.?'
    ),
    "PERSON": re.compile(
        r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'
        rf'|\b(?:{_COMMON_FIRST_NAMES})\s+[A-Z][a-z]+\b'
    ),
}


def scrub(text: str) -> ScrubResult:
    """Replace detected PII entities with typed placeholders.

    Args:
        text: Raw text that may contain PII.

    Returns:
        ScrubResult with the scrubbed text and a count-only summary.
        The original values are not retained anywhere.
    """
    summary = {k: 0 for k in _PATTERNS}
    result = text
    for entity_type, pattern in _PATTERNS.items():
        result, n = pattern.subn(f'[{entity_type}]', result)
        summary[entity_type] = n
    total = sum(summary.values())
    return ScrubResult(
        scrubbed_text=result,
        summary={**summary, "total_entities_scrubbed": total},
        was_scrubbed=total > 0,
    )
