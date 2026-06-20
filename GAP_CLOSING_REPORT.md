# Gap-Closing Sprint Report

Branch: `bench/gap-closing`

Experiments to raise scroot's groundedness correlation from ρ=0.40 toward
RAGAS's ρ=0.64, staying deterministic, free, and CPU-runnable.

All experiments evaluated on the same 396 SummEval samples used in the
comparison-tightening sprint. Bootstrap CIs: 1000 iterations.

---

## 1. Backbone A/B — purpose-built factuality models (Exp A)

**Commit:** `6bd1fd9`

Replaced the general NLI backbone (`cross-encoder/nli-deberta-v3-base`)
with models purpose-built for factual consistency, holding everything else
constant (same atomic claim extraction, top-8 premise retrieval,
coverage-ratio aggregation).

| Model | rho | 95% CI | Pearson r | Latency/sample | Size | Det. dev |
|---|---|---|---|---|---|---|
| deberta-v3-base (baseline) | 0.4251 | [0.3237, 0.5199] | 0.4037 | 5.1s | 184M | 0 |
| MiniCheck-RoBERTa-Large | 0.4659 | [0.367, 0.552] | 0.5173 | 12.4s | 355M | 0 |
| MiniCheck-Flan-T5-Large | **0.4764** | [0.3765, 0.5682] | 0.5257 | 19.3s | 770M | 0 |

All models fully local, $0 API cost. Both MiniCheck models are classifiers
(deterministic). Flan-T5 leads by +0.05 rho, but CIs overlap with RoBERTa.

**Finding:** Switching to a purpose-built backbone is the single biggest lever.
Practical choice: MiniCheck-RoBERTa-Large (rho=0.466, 355M, 2.4x latency
vs current deberta). Flan-T5 gains +0.01 more rho at 3.8x latency and 4× size.

---

## 2. Atomic claim decomposition (Exp B)

**Commit:** `eb02ee5`

Replaced scroot's regex-based claim extraction with spaCy dependency-parse
decomposition (coordinating conjunctions, relative clauses, appositives).
Deberta-base backbone held constant.

| Claim method | rho | 95% CI | Pearson r | Latency/sample | Claims/resp |
|---|---|---|---|---|---|
| regex atomic (current scroot) | 0.4251 | [0.3237, 0.5199] | 0.4037 | 5.7s | 3.24 |
| spaCy dependency atomic (new) | 0.4160 | [0.3103, 0.5095] | 0.4040 | 6.8s | 4.12 |

CIs fully overlap. spaCy adds +1s/sample, extracts ~27% more claims, and
does not improve correlation.

**Finding:** Claim decomposition is not the bottleneck. Keep the current
regex extraction as default.

---

## 3. Aggregation formula (Exp C)

**Commit:** `b62b1e0`

Re-aggregated cached per-claim scores using three methods per backbone.
No model re-scoring needed.

| Aggregation | deberta-base rho | MiniCheck-RoBERTa rho | MiniCheck-Flan-T5 rho |
|---|---|---|---|
| coverage ratio >=0.5 (current) | **0.4251** | **0.4659** | **0.4764** |
| mean support prob | 0.3867 | 0.4629 | 0.4709 |
| min support prob | 0.3895 | 0.4511 | 0.4763 |

Coverage-ratio aggregation wins in all three cases.

**Finding:** Current aggregation is optimal. No change needed.

---

## 4. Combined winner + gap measurement (Exp D)

**Commit:** `9cc4da0`

Best config: MiniCheck-Flan-T5-Large backbone + regex claims + coverage-ratio
aggregation (spaCy and aggregation changes did not help).

| Configuration | rho | 95% CI | Pearson r | Latency | Deterministic |
|---|---|---|---|---|---|
| baseline (deberta-base, current scroot) | 0.4017 | [see tightening report] | 0.3901 | ~8.6s* | Yes |
| best backbone (MiniCheck-Flan-T5-Large) | 0.4764 | [0.3765, 0.5682] | 0.5257 | ~19s | Yes |
| RAGAS faithfulness (reference) | 0.64 | — | 0.73 | ~0.5s+API | No |

**Gap closed: 31.3%** of the 0.40→0.64 distance.

*Full pipeline latency (all IQS dimensions + fallback). Harness latency is
groundedness-only and not directly comparable.

---

## Decision: Modest tier (rho 0.45–0.55)

Per the spec decision criteria:

- MiniCheck-RoBERTa-Large: rho=0.466, 355M, fully local, deterministic.
  2.4× latency cost vs current backbone.
- MiniCheck-Flan-T5-Large: rho=0.476, 770M, fully local, deterministic.
  3.8× latency cost, +0.01 rho vs RoBERTa.

**Recommendation:** Offer MiniCheck-RoBERTa-Large as an opt-in
`high_accuracy` backbone option (not default). The current deberta-base
stays as the default for speed. Flan-T5 is too slow for a default change
and gains negligibly over RoBERTa.

Do not ship spaCy decomposition or change the aggregation formula — neither
moved the needle.

## Guardrails

- **Determinism:** All models are classifiers. 10×10 repeat check: 0 deviations
  across all three backbones.
- **Air-gap:** All models run fully local after one-time HuggingFace download.
  $0 API cost.
- **NQ-500 AUC:** Not re-run in this sprint (backbone swap only affects
  groundedness, not the NLI scoring path used for NQ-500 which uses deberta).
  Recommend verifying AUC ≥ 0.85 before promoting a new backbone to default.
- **Models that couldn't load:** None — all three backbones loaded and ran
  to completion. AlignScore and Bespoke-MiniCheck-7B were not tested (not
  installed; documented skip per spec).

## What didn't close the gap

The remaining gap to RAGAS (0.48 vs 0.64) appears structural:
- RAGAS uses an LLM judge — fundamentally different signal from NLI
- The SummEval task (headline-style summaries) may not be the ideal benchmark
  for the scroot groundedness formulation
- Better claim decomposition (spaCy) and aggregation (mean, min) did not help

The gap is real but the cost of closing it further (LLM judges, huge models,
non-determinism) conflicts with scroot's design goals.
