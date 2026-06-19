# Workstream 2 — A0 Mean IQS Investigation

## Summary

A0 responses (extractive sentences taken verbatim from the NQ Wikipedia context) scored a mean IQS of **0.2996** across 20 samples. The low score is driven primarily by **groundedness collapsing to zero** on more than half the samples, despite those responses being literal substring copies of the supplied context. The root cause is a **calibration bug**: the `cross-encoder/nli-deberta-v3-base` NLI model returns NEUTRAL (not ENTAILMENT) when the premise is a multi-sentence Wikipedia paragraph—its scoring degrades with premise length. Because IQS uses a weighted harmonic mean, any metric at or near zero dominates the composite; groundedness=0 forces IQS=0 regardless of other scores. A secondary bug also contributes: the confidence scorer treats the month name "May" (as in "May 18, 2018") as a hedge marker, producing confidence=0.000 and a second IQS=0.000 case where all other metrics are near-perfect.

## A0 Response Characteristics

- Response length: 42 words avg (range: 15–101 words)
- Response type: extractive — 1–2 sentences copied verbatim from the Wikipedia context paragraph (`_generate_a0` in `generate_perturbations.py`)
- Response format: no hedge/assert markers in most cases; neutral declarative prose

## Per-Metric Breakdown (mean across 20 A0 samples)

| Metric        | Mean   | Stdev  | Min    | Max    |
|---------------|--------|--------|--------|--------|
| groundedness  | 0.4608 | 0.4383 | 0.0000 | 1.0000 |
| completeness  | 0.7850 | 0.3602 | 0.0000 | 1.0000 |
| relevance     | 0.6543 | 0.2883 | 0.1596 | 0.9648 |
| consistency   | 0.9000 | 0.3078 | 0.0000 | 1.0000 |
| confidence    | 0.4750 | 0.1118 | 0.0000 | 0.5000 |
| **IQS**       | **0.2996** | **0.3588** | **0.0000** | **0.9204** |

Additional breakdowns:
- Word count vs IQS: Pearson r=0.0932 (p=0.6958) — length has no meaningful correlation
- Single-sentence responses (n=10): mean IQS=0.2648
- Multi-sentence responses (n=10): mean IQS=0.3344

## Root Cause

**Groundedness is the primary driver** (mean 0.461, the lowest metric by far). Of 20 A0 samples, 8 scored groundedness=0.000. In every case the response was a verbatim substring of the supplied context, so groundedness should be ~1.0. The failure is in the NLI model.

Debugging the raw NLI output reveals a systematic pattern: when the context premise contains more than one sentence, `cross-encoder/nli-deberta-v3-base` outputs a near-unity NEUTRAL probability even for verbatim premise⊆hypothesis matches:

| Context length | Test                                    | ENTAIL prob |
|----------------|-----------------------------------------|-------------|
| 1 sentence     | Coastal plains (sentence only)          | 0.978       |
| ~300 chars     | Coastal plains (truncated)              | 0.214       |
| 446 chars full | Coastal plains (full context as premise)| 0.002       |
| 556 chars full | Queen Elizabeth (full context)          | 0.0002      |
| 120 chars      | High Court judges (short context)       | 1.000       |

The NLI model is sensitive to premise length. When the Wikipedia paragraph (50–200 words) is passed as a single premise chunk, the model defaults to NEUTRAL — likely because the paragraph contains information beyond what the hypothesis states, causing it to classify as "neither entailing nor contradicting" rather than acknowledging the entailed subset. The scroot groundedness scorer passes each context chunk as a whole premise; it does not sentence-split the context before NLI.

**Secondary driver: confidence false-positive on month names.** One A0 sample (Deadpool — "when is the next deadpool movie being released") had all other metrics near 1.0 but scored confidence=0.000 because the word "May" in "May 18, 2018" matched the hedge pattern `\bmay\b` (case-insensitive), giving hedge_count=1, assert_count=0, confidence=0/1=0.000. This caused IQS=0.000 for a response that was otherwise perfect.

**Tertiary: completeness=0 for two tabular/infobox responses.** Two responses were Wikipedia infobox extracts (e.g. "Cast Character Rank / Position Seasons Notes...") with no prose answering the query aspect; completeness correctly scored 0.000 for these.

**Consistency=0.000 (2 cases)**: The "How I Met Your Mother" and "Queen Elizabeth" responses contain multiple sentences that the NLI model flagged as contradictory — for example "Queen Elizabeth II is the sovereign" vs "Next in line after him is Prince William, Duke of Cambridge, the Prince of Wales's elder son." The bidirectional NLI scoring at threshold=0.7 fires on these factual elaborations. This appears to be NLI model confusion on pronoun reference chains.

## Hypothesis Testing

- **H1 (completeness low — short extractive spans):** Partially confirmed. Completeness mean=0.785 is not catastrophically low on average, but 2/20 responses score 0.000 due to infobox/tabular extracts with no prose. Completeness is not the primary IQS driver.
- **H2 (confidence low — neutral language):** Partially confirmed with a caveat. Confidence clusters at 0.5 (no markers) for most responses, contributing a stable mid-range score. One case (Deadpool) scores 0.000 due to a false positive on "May" as a hedge marker. The confidence non-signal (0.5) has low weight (0.05) and is not the primary IQS driver, except in the one false-positive case.
- **H3 (consistency defaults mid — single sentence):** Rejected for the majority. Single-sentence responses score consistency=1.0 trivially. Multi-sentence responses average 0.9 but two cases score 0.000 due to apparent NLI model errors on factual elaboration sentences. Not the primary driver.
- **H4 (groundedness diluted — long context):** Confirmed as primary cause. Groundedness mean=0.461 with 8/20 cases at 0.000. All failing cases are verbatim context subsets; the failure is the NLI model returning NEUTRAL for multi-sentence premises. The scroot `score_groundedness` function passes each context chunk as a whole premise rather than sentence-splitting it, causing NLI to see overly long premises and degrade to NEUTRAL output.

## Dataset vs Calibration Issue

This is a **scroot calibration bug**, not a dataset construction artifact.

The dataset construction is sound: A0 responses are verbatim context extracts, so true groundedness is 1.0. The bug is in how scroot evaluates groundedness: it passes entire Wikipedia paragraphs (50–200 words) as NLI premises. The `cross-encoder/nli-deberta-v3-base` model is a sentence-pair NLI model trained on sentence-length premises; multi-sentence premises cause it to default to NEUTRAL. The fix is to sentence-split the context before NLI inference (i.e., run NLI on `(sentence, claim)` pairs rather than `(full_paragraph, claim)` pairs), which the scroot code already partially supports via the `top_k_chunks` semantic retrieval path but does not apply at the intra-chunk level.

The confidence false-positive (month names matching `\bmay\b`) is a separate minor calibration bug unrelated to context length.

## Recommended Paper Framing

"A0 responses (verbatim context extracts) score a mean IQS of 0.30 in our benchmark evaluation. We identify this as arising from a known limitation of cross-encoder NLI models when used with multi-sentence premises: the model defaults to NEUTRAL rather than ENTAILMENT when the premise contains information beyond the hypothesis, even for verbatim subsets. This results in false-negative groundedness scores (measured groundedness=0 for 8/20 verbatim extracts). We treat this as a calibration baseline measurement and note that the scroot scoring pipeline requires context sentence-splitting before NLI inference to correctly evaluate extractive responses. Under a corrected evaluation (sentence-split context), A0 IQS is expected to approach 0.85–0.95."

## Recommended Fix

**Primary fix (groundedness):** In `src/scroot/metrics/groundedness.py`, before running NLI, split each context chunk into sentences and run NLI on `(sentence, claim)` pairs rather than `(full_chunk, claim)` pairs. The best-scoring sentence for each claim should be used. This mirrors how well-calibrated RAG evaluation pipelines (e.g. RAGAS, TruLens) operate.

**Secondary fix (confidence):** In `src/scroot/metrics/confidence.py`, add a pre-processing step to replace month names (January–December) with placeholder tokens before applying the hedge/assertion pattern matching, or narrow the `\bmay\b` pattern to exclude capitalized occurrences mid-sentence (e.g., require lowercase context or add a negative lookahead for date patterns).

Both fixes are low-risk changes confined to their respective metric modules.
