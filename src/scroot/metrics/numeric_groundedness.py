"""Numeric grounding verifier: detect numeric hallucinations missed by semantic NLI.

Four-layer pipeline:
  1. Regex extraction: identify all numeric claims in the response.
  2. Unit normalization: canonicalize values so 1.5 km == 1500 m.
  3. Claim-level grounding: is each number present or consistent with context?
  4. NLI integration: optional supplementary signal for ambiguous claims.

Numeric hallucinations are the #1 RAG failure mode and are routinely missed by
semantic NLI because NLI models encode semantic similarity, not arithmetic equality.
This verifier operates on the numeric surface form and is complementary to the
groundedness NLI pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Layer 1: Numeric extraction
# ---------------------------------------------------------------------------

# Scan for "number" and look for adjacent unit text.
_CLAIM_RE = re.compile(
    r"""
    (?:[$€£¥]\s*)?                       # optional currency prefix
    (?:\d{1,3}(?:,\d{3})*(?:\.\d+)?     # comma-thousands
      |\d+(?:\.\d+)?)                    # or plain decimal
    (?:\s*(?:%
          |km|mi(?:le|les)?|m(?!\w)|cm|mm|ft|feet|in(?:ch(?:es)?)?|yd|yard(?:s)?
          |second(?:s)?|sec(?:s)?|minute(?:s)?|min(?:s)?|hour(?:s)?|hr(?:s)?
          |day(?:s)?|week(?:s)?|month(?:s)?|year(?:s)?|yr(?:s)?
          |kg|g(?!\w)|mg|lb(?:s)?|pound(?:s)?|oz|ounce(?:s)?|ton(?:ne)?(?:s)?
          |l(?!\w)|ml|liter(?:s)?|litre(?:s)?|gal(?:lon(?:s)?)?
          |tb|gb|mb|kb|b(?!\w)|byte(?:s)?|kilobyte(?:s)?|megabyte(?:s)?
          |gigabyte(?:s)?|terabyte(?:s)?
          |mph|kph|[$€£¥]|usd|eur|gbp|jpy
    )(?!\w))?
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass
class NumericClaim:
    """A single numeric claim extracted from text."""

    raw: str
    value: float
    unit: str
    unit_family: str
    canonical: float
    source_text: str


# ---------------------------------------------------------------------------
# Layer 2: Unit normalization
# ---------------------------------------------------------------------------

_UNIT_MAP: dict[str, tuple[str, str, float]] = {
    # (canonical_unit, family, factor_to_base)
    # Distance (base unit: meters)
    "m": ("m", "distance", 1.0),
    "meter": ("m", "distance", 1.0), "meters": ("m", "distance", 1.0),
    "metre": ("m", "distance", 1.0), "metres": ("m", "distance", 1.0),
    "km": ("m", "distance", 1000.0),
    "kilometer": ("m", "distance", 1000.0), "kilometers": ("m", "distance", 1000.0),
    "kilometre": ("m", "distance", 1000.0), "kilometres": ("m", "distance", 1000.0),
    "cm": ("m", "distance", 0.01),
    "centimeter": ("m", "distance", 0.01), "centimeters": ("m", "distance", 0.01),
    "mm": ("m", "distance", 0.001),
    "millimeter": ("m", "distance", 0.001), "millimeters": ("m", "distance", 0.001),
    "mile": ("m", "distance", 1609.344), "miles": ("m", "distance", 1609.344),
    "mi": ("m", "distance", 1609.344),
    "ft": ("m", "distance", 0.3048), "foot": ("m", "distance", 0.3048),
    "feet": ("m", "distance", 0.3048),
    "in": ("m", "distance", 0.0254), "inch": ("m", "distance", 0.0254),
    "inches": ("m", "distance", 0.0254),
    "yd": ("m", "distance", 0.9144), "yard": ("m", "distance", 0.9144),
    "yards": ("m", "distance", 0.9144),
    # Time (base unit: seconds)
    "second": ("s", "time", 1.0), "seconds": ("s", "time", 1.0),
    "sec": ("s", "time", 1.0), "secs": ("s", "time", 1.0),
    "minute": ("s", "time", 60.0), "minutes": ("s", "time", 60.0),
    "min": ("s", "time", 60.0), "mins": ("s", "time", 60.0),
    "hour": ("s", "time", 3600.0), "hours": ("s", "time", 3600.0),
    "hr": ("s", "time", 3600.0), "hrs": ("s", "time", 3600.0),
    "day": ("s", "time", 86400.0), "days": ("s", "time", 86400.0),
    "week": ("s", "time", 604800.0), "weeks": ("s", "time", 604800.0),
    "month": ("s", "time", 2592000.0), "months": ("s", "time", 2592000.0),
    "year": ("s", "time", 31536000.0), "years": ("s", "time", 31536000.0),
    "yr": ("s", "time", 31536000.0), "yrs": ("s", "time", 31536000.0),
    # Mass (base unit: grams)
    "g": ("g", "mass", 1.0), "gram": ("g", "mass", 1.0), "grams": ("g", "mass", 1.0),
    "kg": ("g", "mass", 1000.0),
    "kilogram": ("g", "mass", 1000.0), "kilograms": ("g", "mass", 1000.0),
    "mg": ("g", "mass", 0.001),
    "milligram": ("g", "mass", 0.001), "milligrams": ("g", "mass", 0.001),
    "lb": ("g", "mass", 453.592), "lbs": ("g", "mass", 453.592),
    "pound": ("g", "mass", 453.592), "pounds": ("g", "mass", 453.592),
    "oz": ("g", "mass", 28.3495), "ounce": ("g", "mass", 28.3495),
    "ounces": ("g", "mass", 28.3495),
    "ton": ("g", "mass", 1_000_000.0), "tons": ("g", "mass", 1_000_000.0),
    "tonne": ("g", "mass", 1_000_000.0), "tonnes": ("g", "mass", 1_000_000.0),
    # Volume (base unit: liters)
    "l": ("l", "volume", 1.0), "liter": ("l", "volume", 1.0),
    "liters": ("l", "volume", 1.0), "litre": ("l", "volume", 1.0),
    "litres": ("l", "volume", 1.0),
    "ml": ("l", "volume", 0.001),
    "milliliter": ("l", "volume", 0.001), "milliliters": ("l", "volume", 0.001),
    "gal": ("l", "volume", 3.78541), "gallon": ("l", "volume", 3.78541),
    "gallons": ("l", "volume", 3.78541),
    # Data (base unit: bytes)
    "b": ("b", "data", 1.0), "byte": ("b", "data", 1.0), "bytes": ("b", "data", 1.0),
    "kb": ("b", "data", 1024.0), "kilobyte": ("b", "data", 1024.0),
    "kilobytes": ("b", "data", 1024.0),
    "mb": ("b", "data", 1_048_576.0), "megabyte": ("b", "data", 1_048_576.0),
    "megabytes": ("b", "data", 1_048_576.0),
    "gb": ("b", "data", 1_073_741_824.0), "gigabyte": ("b", "data", 1_073_741_824.0),
    "gigabytes": ("b", "data", 1_073_741_824.0),
    "tb": ("b", "data", 1_099_511_627_776.0), "terabyte": ("b", "data", 1_099_511_627_776.0),
    "terabytes": ("b", "data", 1_099_511_627_776.0),
    # Percentage
    "%": ("%", "percentage", 1.0),
    # Currency (no cross-currency conversion; match by family only)
    "$": ("currency", "currency", 1.0), "€": ("currency", "currency", 1.0),
    "£": ("currency", "currency", 1.0), "¥": ("currency", "currency", 1.0),
    "usd": ("currency", "currency", 1.0), "eur": ("currency", "currency", 1.0),
    "gbp": ("currency", "currency", 1.0), "jpy": ("currency", "currency", 1.0),
}


def _parse_float(text: str) -> float | None:
    cleaned = re.sub(r'[$€£¥]', '', text).replace(",", "").strip()
    m = re.match(r'^[\d.]+', cleaned)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _normalize(unit_str: str) -> tuple[str, str, float]:
    """Return (canonical_unit, family, factor) for a unit string."""
    key = unit_str.strip().lower()
    if key in _UNIT_MAP:
        return _UNIT_MAP[key]
    # Try without trailing 's' (naive plural)
    if key.endswith("s") and key[:-1] in _UNIT_MAP:
        return _UNIT_MAP[key[:-1]]
    return ("", "unitless", 1.0)


def _extract_claims(text: str) -> list[NumericClaim]:
    """Extract all numeric claims from text, splitting on sentences first."""
    from ..text_utils import split_sentences
    claims: list[NumericClaim] = []
    sentences = split_sentences(text) if text.strip() else [text]

    for sent in sentences:
        for m in _CLAIM_RE.finditer(sent):
            raw = m.group().strip()
            if not raw or not re.search(r'\d', raw):
                continue
            value = _parse_float(raw)
            if value is None:
                continue

            # Identify the unit portion (everything after the number)
            num_end = re.match(
                r'[$€£¥]?\s*(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)',
                raw,
            )
            unit_part = raw[num_end.end():].strip() if num_end else ""

            # Currency prefix without explicit unit
            has_currency_prefix = bool(re.match(r'^[$€£¥]', raw))
            if not unit_part and has_currency_prefix:
                unit_part = raw[0]

            canonical_unit, family, factor = _normalize(unit_part) if unit_part else ("", "unitless", 1.0)

            claims.append(NumericClaim(
                raw=raw,
                value=value,
                unit=canonical_unit,
                unit_family=family,
                canonical=value * factor,
                source_text=sent,
            ))

    return claims


# ---------------------------------------------------------------------------
# Layer 3: Claim-level grounding
# ---------------------------------------------------------------------------

_REL_TOLERANCE = 0.02  # 2% relative tolerance


def _claim_grounded(claim: NumericClaim, ctx_claims: list[NumericClaim], tolerance: float) -> bool:
    for ctx in ctx_claims:
        if ctx.unit_family != claim.unit_family:
            continue
        denom = max(abs(claim.canonical), abs(ctx.canonical), 1e-9)
        if abs(claim.canonical - ctx.canonical) / denom <= tolerance:
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 4 + public API
# ---------------------------------------------------------------------------

def score_numeric_groundedness(
    response: str,
    context: list[str] | None,
    *,
    nli_model: str | None = None,
    device: str = "cpu",
    tolerance: float = _REL_TOLERANCE,
) -> tuple[float | None, dict]:
    """Score how well numeric claims in the response are grounded in context.

    Args:
        response: LLM response text.
        context: Grounding context chunks. ``None`` → returns ``(None, {})``.
        nli_model: Optional NLI model for supplementary signal. When a numeric
            claim is not found by matching, the containing sentence is checked
            via NLI against each context chunk; entailment counts as grounded.
        device: ``"cpu"`` or ``"cuda"``.
        tolerance: Relative tolerance for numeric match. Default 0.02 (2%).

    Returns:
        ``(score, details)`` where score is in ``[0, 1]`` or ``None`` when
        context is absent, and details contains the per-claim breakdown.
    """
    if context is None:
        return None, {"note": "no context — numeric grounding skipped"}

    response_claims = _extract_claims(response)
    if not response_claims:
        return 1.0, {"note": "no numeric claims found in response", "claims": []}

    ctx_text = " ".join(context)
    ctx_claims = _extract_claims(ctx_text)

    grounded_count = 0
    claim_details: list[dict] = []

    nli_m = None
    if nli_model is not None:
        from ..models import get_nli_model
        nli_m = get_nli_model(nli_model, device=device)

    for claim in response_claims:
        grounded_by_num = _claim_grounded(claim, ctx_claims, tolerance)

        grounded_by_nli = False
        if not grounded_by_num and nli_m is not None:
            from ._utils import softmax
            pairs = [(claim.source_text, chunk) for chunk in context if chunk.strip()]
            if pairs:
                raw_scores = nli_m.predict(pairs)
                for raw_s in raw_scores:
                    probs = softmax(raw_s)
                    if float(probs[2]) >= 0.5:  # label 2 = entailment
                        grounded_by_nli = True
                        break

        is_grounded = grounded_by_num or grounded_by_nli
        if is_grounded:
            grounded_count += 1

        claim_details.append({
            "raw": claim.raw,
            "value": claim.value,
            "unit": claim.unit,
            "unit_family": claim.unit_family,
            "canonical_value": claim.canonical,
            "grounded_by_numeric": grounded_by_num,
            "grounded_by_nli": grounded_by_nli,
            "grounded": is_grounded,
        })

    score = grounded_count / len(response_claims)
    details: dict = {
        "total_numeric_claims": len(response_claims),
        "grounded_claims": grounded_count,
        "ungrounded_claims": len(response_claims) - grounded_count,
        "claims": claim_details,
        "tolerance": tolerance,
    }
    if nli_model:
        details["nli_supplemented"] = True

    return round(score, 4), details
