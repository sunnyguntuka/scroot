"""Sentence splitting and text preprocessing utilities.

NLTK is NOT downloaded at runtime (H-6). The regex splitter is used by
default. If NLTK is already installed and punkt_tab has been downloaded
(via scroot.setup_nltk()), it is used for improved sentence boundary
detection. If NLTK is unavailable or punkt_tab is missing, the regex
fallback is used silently.

To opt into NLTK-backed splitting, run once after installation:
    python -c "import scroot; scroot.setup_nltk()"
"""

import re

# Patterns that split compound sentences into atomic sub-claims.
# Matches ", and/but/or/while/whereas/although/though/yet " conjunctions
# that join independent clauses.
_COMPOUND_CONJ = re.compile(
    r",\s+(?:and|but|or|nor|while|whereas|although|though|yet)\s+",
    re.IGNORECASE,
)

_NON_CLAIM_PATTERNS = [
    r"^\s*(hi|hello|hey|thanks|thank you)",
    r"\?\s*$",
    r"^\s*(sure|okay|of course|certainly)",
]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Tries NLTK sent_tokenize if punkt_tab data is already present on disk.
    Falls back to a regex splitter without any network access.

    Args:
        text: Input text string.

    Returns:
        List of sentence strings, stripped and non-empty.
    """
    if not text or not text.strip():
        return []

    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    return [s.strip() for s in sentences if s.strip()]


def _is_non_claim(sent: str) -> bool:
    for pattern in _NON_CLAIM_PATTERNS:
        if re.search(pattern, sent, re.IGNORECASE):
            return True
    return False


def _split_into_subclaims(sentence: str) -> list[str]:
    """Split a compound sentence into atomic sub-claims.

    Handles:
    - Semicolons:  "A; B" -> ["A", "B"]
    - Compound conjunctions:  "A, and B" -> ["A", "B"]
    """
    # Step 1: split on semicolons
    parts = re.split(r";\s*", sentence)

    result = []
    for part in parts:
        # Step 2: split on ", and/but/or..." joining independent clauses
        subparts = _COMPOUND_CONJ.split(part)
        result.extend(subparts)

    return [p.strip() for p in result if p.strip()]


def extract_claims(text: str) -> list[str]:
    """Extract individual factual claims from text.

    A claim is a sentence or clause that makes a factual assertion.
    Filters out questions, greetings, hedging-only sentences.

    Args:
        text: Input text string.

    Returns:
        List of claim strings.
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    claims = []
    for sent in sentences:
        if not _is_non_claim(sent) and len(sent.split()) >= 3:
            claims.append(sent)
    return claims


def extract_atomic_claims(text: str) -> list[str]:
    """Extract atomic (sub-sentence) factual claims from text.

    Splits compound sentences into individual verifiable facts before
    filtering. This gives finer-grained groundedness scoring: a response
    with 9 correct claims and 1 wrong claim scores ~0.9 instead of 0.

    Compared to extract_claims():
    - "Coffee was found in Yemen, and it spread to Arabia."
      extract_claims()       -> 1 claim  (full sentence)
      extract_atomic_claims() -> 2 claims ["Coffee was found in Yemen",
                                           "it spread to Arabia"]

    Args:
        text: Input text string.

    Returns:
        List of atomic claim strings, each >= 4 words.
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    atomic: list[str] = []

    for sent in sentences:
        if _is_non_claim(sent):
            continue
        sub_claims = _split_into_subclaims(sent)
        for sc in sub_claims:
            if len(sc.split()) >= 4:
                atomic.append(sc)

    return atomic if atomic else extract_claims(text)
