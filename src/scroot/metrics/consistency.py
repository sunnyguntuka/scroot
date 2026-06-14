"""Consistency metric: does the response contradict itself?

Bidirectional NLI: for each sentence pair (A, B), run NLI in both
directions and take the maximum contradiction probability.

  forward:  NLI(A premise, B hypothesis) - does A contradict B?
  backward: NLI(B premise, A hypothesis) - does B contradict A?

Taking the max catches asymmetric contradictions that a one-direction
pass misses, e.g. "The service is fast" vs "Response times are slow."
"""

from __future__ import annotations

import warnings
from itertools import combinations

from ..models import get_nli_model
from ..text_utils import split_sentences
from ._utils import softmax

LABEL_CONTRADICTION = 0


def score_consistency(
    response: str,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    device: str = "cpu",
    contradiction_threshold: float = 0.7,
    max_sentences: int = 25,
    bidirectional: bool = True,
) -> tuple[float, dict]:
    """Score the internal consistency of a response.

    Checks sentence pairs for contradictions using bidirectional NLI.
    Score = 1.0 - (fraction of contradictory pairs), clamped to [0, 1].

    Args:
        response: The LLM response text.
        nli_model: NLI cross-encoder model name or pre-instantiated instance.
        device: "cpu" or "cuda".
        contradiction_threshold: Minimum contradiction probability to flag a
            pair as contradictory. Default 0.7.
        max_sentences: Maximum sentences evaluated (H-4 cap). Longer
            responses use first/last half. Default 25.
        bidirectional: If True (default), run NLI in both A→B and B→A
            directions and take the max contradiction probability.
            Catches asymmetric contradictions missed by one-direction NLI.

    Returns:
        Tuple of (score, details_dict).
    """
    if not response or not response.strip():
        return 1.0, {"note": "empty response, consistency trivially 1.0"}

    model = get_nli_model(nli_model, device=device)
    sentences = split_sentences(response)

    if len(sentences) <= 1:
        return 1.0, {"note": "single sentence, consistency trivially 1.0"}

    truncated = False
    if len(sentences) > max_sentences:
        half = max_sentences // 2
        sentences = sentences[:half] + sentences[-half:]
        truncated = True
        warnings.warn(
            f"Response has more than {max_sentences} sentences; "
            f"consistency scored on first/last {half} sentences only (H-4).",
            stacklevel=3,
        )

    pairs = list(combinations(range(len(sentences)), 2))
    if not pairs:
        return 1.0, {"note": "no pairs to check"}

    if bidirectional:
        # Build forward and backward pairs in one batch
        forward_pairs = [(sentences[i], sentences[j]) for i, j in pairs]
        backward_pairs = [(sentences[j], sentences[i]) for i, j in pairs]
        all_pairs = forward_pairs + backward_pairs
        all_scores = model.predict(all_pairs)
        n = len(pairs)
        fwd_scores = all_scores[:n]
        bwd_scores = all_scores[n:]
    else:
        fwd_scores = model.predict([(sentences[i], sentences[j]) for i, j in pairs])
        bwd_scores = None

    contradictions = []
    for idx, (i, j) in enumerate(pairs):
        fwd_probs = softmax(fwd_scores[idx])
        fwd_cp = float(fwd_probs[LABEL_CONTRADICTION])

        if bidirectional and bwd_scores is not None:
            bwd_probs = softmax(bwd_scores[idx])
            bwd_cp = float(bwd_probs[LABEL_CONTRADICTION])
            contradiction_prob = max(fwd_cp, bwd_cp)
        else:
            contradiction_prob = fwd_cp

        if contradiction_prob >= contradiction_threshold:
            contradictions.append({
                "sentence_a": sentences[i],
                "sentence_b": sentences[j],
                "contradiction_prob": round(contradiction_prob, 4),
            })

    raw_score = 1.0 - (len(contradictions) / len(pairs))
    consistency_score = max(0.0, min(1.0, raw_score))

    details: dict = {
        "total_pairs": len(pairs),
        "contradictions_found": len(contradictions),
        "contradictions": contradictions,
        "bidirectional": bidirectional,
    }
    if truncated:
        details["truncated"] = True
        details["note"] = (
            f"Scored on {max_sentences} of the original sentences "
            f"(first/last {max_sentences // 2})"
        )

    return consistency_score, details
