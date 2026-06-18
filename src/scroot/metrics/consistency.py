"""Consistency metric: does the response contradict itself?

Bidirectional NLI: for each sentence pair (A, B), run NLI in both
directions and take the maximum contradiction probability.

  forward:  NLI(A premise, B hypothesis) - does A contradict B?
  backward: NLI(B premise, A hypothesis) - does B contradict A?

Taking the max catches asymmetric contradictions that a one-direction
pass misses, e.g. "The service is fast" vs "Response times are slow."

**Computational cost:** NLI is O(n²) in sentence count. For a 25-sentence
response, bidirectional mode runs ~300 NLI calls (10–30s on CPU). Controls:

- ``bidirectional=False``: halves NLI calls with a small accuracy trade-off.
  Misses asymmetric contradictions where only one direction fires.
- ``max_sentences`` (default 25): sentences above this cap are truncated to
  first/last N/2. Keeps the ceiling at 300 calls.
- ``pair_sample_size``: when the sentence count exceeds 20, randomly sample
  this many pairs instead of evaluating all C(n,2). Trades completeness for
  speed on verbose responses (e.g. sample 100 of ~190 pairs for 20 sentences).
"""

from __future__ import annotations

import random
import warnings
from itertools import combinations

from ..models import get_nli_model
from ..text_utils import split_sentences
from ._utils import softmax

LABEL_CONTRADICTION = 0

# Pair count above which we switch to sampling to bound NLI calls.
_PAIR_SAMPLE_THRESHOLD = 20
_DEFAULT_PAIR_SAMPLE_SIZE = 150


def score_consistency(
    response: str,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    device: str = "cpu",
    contradiction_threshold: float = 0.7,
    max_sentences: int = 25,
    bidirectional: bool = True,
    pair_sample_size: int | None = None,
    pair_sample_seed: int | None = 42,
) -> tuple[float, dict]:
    """Score the internal consistency of a response.

    Checks sentence pairs for contradictions using bidirectional NLI.
    Score = 1.0 - (fraction of contradictory pairs), clamped to [0, 1].

    **Cost note:** bidirectional NLI is O(n²). For responses with more than
    ``_PAIR_SAMPLE_THRESHOLD`` sentences (after truncation), pairs are
    randomly sampled to ``pair_sample_size`` to keep latency bounded.
    Disable sampling by setting ``pair_sample_size=0``.

    Args:
        response: The LLM response text.
        nli_model: NLI cross-encoder model name or pre-instantiated instance.
        device: ``"cpu"`` or ``"cuda"``.
        contradiction_threshold: Minimum contradiction probability to flag a
            pair as contradictory. Default 0.7.
        max_sentences: Maximum sentences evaluated (H-4 cap). Longer
            responses use first/last N/2. Default 25.
        bidirectional: If ``True`` (default), run NLI in both A→B and B→A
            directions and take the max contradiction probability. Set to
            ``False`` to halve NLI calls at a small accuracy cost.
        pair_sample_size: When the response has more than
            ``_PAIR_SAMPLE_THRESHOLD`` sentences (20), sample this many
            pairs instead of evaluating C(n,2). Default 150. Set to ``0``
            to disable sampling and evaluate all pairs (may be slow).
        pair_sample_seed: Random seed for reproducible pair sampling.
            Default 42. ``None`` → non-deterministic.

    Returns:
        Tuple of ``(score, details_dict)``.
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

    all_pair_indices = list(combinations(range(len(sentences)), 2))
    if not all_pair_indices:
        return 1.0, {"note": "no pairs to check"}

    # Pair sampling for responses above the threshold
    sampled = False
    pairs = all_pair_indices
    effective_sample_size = pair_sample_size if pair_sample_size is not None else _DEFAULT_PAIR_SAMPLE_SIZE
    if (
        len(sentences) > _PAIR_SAMPLE_THRESHOLD
        and effective_sample_size > 0
        and len(all_pair_indices) > effective_sample_size
    ):
        rng = random.Random(pair_sample_seed)
        pairs = rng.sample(all_pair_indices, effective_sample_size)
        sampled = True

    if bidirectional:
        forward_pairs = [(sentences[i], sentences[j]) for i, j in pairs]
        backward_pairs = [(sentences[j], sentences[i]) for i, j in pairs]
        all_pairs_text = forward_pairs + backward_pairs
        all_scores = model.predict(all_pairs_text)
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
        "total_possible_pairs": len(all_pair_indices),
        "contradictions_found": len(contradictions),
        "contradictions": contradictions,
        "bidirectional": bidirectional,
        "pair_sampled": sampled,
    }
    if sampled:
        details["pair_sample_size"] = len(pairs)
        details["note"] = (
            f"Pair sampling active: scored {len(pairs)} of "
            f"{len(all_pair_indices)} possible pairs (sentence count > "
            f"{_PAIR_SAMPLE_THRESHOLD}). Set pair_sample_size=0 to "
            "disable sampling."
        )
    elif truncated:
        details["truncated"] = True
        details["note"] = (
            f"Scored on {max_sentences} of the original sentences "
            f"(first/last {max_sentences // 2})"
        )

    return consistency_score, details
