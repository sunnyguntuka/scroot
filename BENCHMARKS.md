# scroot Benchmarks

**scroot** is an LLM-free, deterministic response quality scorer. This document is the single
source of truth for all benchmark evidence: what was measured, how, the exact numbers, what
failed, what bugs were found, and how to reproduce everything.

**Last updated:** 2026-06-21 · Numbers produced on commits spanning branches
`bench/comparison-tightening` → `bench/gap-closing` → `bench/minicheck-nq500-gate` →
`bench/minicheck-fullpipeline` (latest: `0d670c0`). See §10 for branch-by-branch provenance
and what needs a post-merge re-run on main.

→ Jump to [Reproducibility](#10-reproducibility)

---

## 1. Headline summary

| Capability | scroot (MiniCheck, high-accuracy) | scroot (deberta, fast) | RAGAS (best LLM judge) | DeepEval |
|:---|:---:|:---:|:---:|:---:|
| Hallucination discrimination — NQ-500 AUC (A0 vs A4) | **0.991** | 0.875 | — | — |
| Human correlation — SummEval Spearman ρ | **0.47** | 0.43 | 0.64 | 0.28 |
| Determinism | **100%** | **100%** | No | No |
| Cost per evaluation | **$0.00** | **$0.00** | ~$0.00052 | ~$0.00004 |
| Offline / air-gapped | **Yes** | **Yes** | No | No |
| Latency — p50, full pipeline | ~4.8s | ~3.2s | ~0.5s + API RTT | ~8s |

scroot achieves near-perfect hallucination discrimination (AUC 0.991) and ~73% of the best
LLM judge's human correlation, while being the only option that is deterministic, free, and
offline. LLM-as-judge tools achieve higher human correlation at per-call API cost and without
reproducibility. scroot is the right tool where you need reproducible scores for CI gates,
compliance audits, or production monitoring at scale. LLM judges are better for one-off
research evaluation where correlation is everything and cost is no object.

**Backbone options:**
```python
Auditor()                                                    # deberta-base (fast, default)
Auditor(groundedness_backbone="minicheck-roberta-large")     # MiniCheck-RoBERTa (high-accuracy)
```

---

## 2. What scroot measures

scroot scores every response on five independent quality dimensions, then combines them into a
single composite IQS (Information Quality Score):

| Dimension | What it measures | Method |
|:---|:---|:---|
| **Groundedness** | Is the response faithful to source context? | NLI cross-encoder per atomic claim; coverage-ratio aggregation |
| **Completeness** | Did the response address all parts of the query? | Embedding similarity per query segment |
| **Relevance** | Is the response on-topic? | Cosine similarity query↔response |
| **Consistency** | Does the response contradict itself? | Pairwise NLI between sentences |
| **Confidence** | Is language assertive vs hedged? | Regex pattern matching |

**IQS** is the weighted harmonic mean of all applicable dimensions (weights: groundedness 0.35,
completeness 0.25, relevance 0.20, consistency 0.15, confidence 0.05). The harmonic mean
penalises any zero dimension hard — a response that is grounded but completely off-topic scores
low, correctly.

**Numeric grounding verifier:** numeric claims (prices, dates, percentages) are verified
separately against context before NLI scoring. A verbatim number match bypasses the NLI
uncertain zone.

**Groundedness backbones:**
- `deberta-base` (default): `cross-encoder/nli-deberta-v3-base`, 184M, 3-class NLI with
  softmax; fast, deterministic, $0.
- `minicheck-roberta-large` (opt-in): `lytang/MiniCheck-RoBERTa-Large`, 355M, binary support
  classifier purpose-built for factual consistency; higher correlation and discrimination,
  1.75× latency.

---

## 3. Methodology

### Datasets

**SummEval** (Fabbri et al. 2021): 100 CNN/DailyMail articles × 16 system summaries = 1,600
annotated samples. Expert human annotations for consistency (faithfulness), relevance,
coherence, and fluency. The faithfulness target throughout this document is the mean expert
**consistency** score. The **396-sample subset** used in all competitor comparisons is the set
of samples DeepEval successfully scored (4 of 400 stratified samples lost to API timeouts;
scroot and RAGAS were aligned to the same 396 pairs, 0 human-consistency mismatches across
sources).

**NQ-500**: 500 Google Natural Questions examples × 5 perturbation levels = 2,500 scored
records. Perturbation levels:

| Level | Description |
|:---:|:---|
| A0 | Correct answer extracted from Wikipedia context — fully grounded |
| A1 | Same answer with added epistemic hedging ("reportedly…") |
| A2 | One grounded sentence + one fabricated sentence |
| A3 | Fully fabricated answer — topically plausible but unsupported |
| A4 | Completely off-topic response |

This is a discrimination test, not a human-correlation test. Metric: AUC(A0 vs A4) —
probability that a grounded response outscores a fabricated one. Gate: ≥ 0.85.

### Hardware

Intel Core i7 CPU, single thread, Windows 11, 32 GB RAM. All model inference is CPU-only
(no GPU). Latency figures are warm-cache (models pre-loaded); first-call figures include
model-weight loading.

### Statistical methods

**Spearman ρ** and **Pearson r** with two-sided p-values (scipy). **Bootstrap 95% CIs**:
1,000 paired resamples, seed 1234, percentile method. The paired-difference CI is the powered
test of record for A/B comparisons — per-model CIs overlap even when the paired difference is
significant because both models score the same samples.

### What "deterministic" means

Same input → identical output, bit-for-bit, across runs. This is a stronger guarantee than
"pinned LLM" (which still varies with server batching and temperature). scroot uses no
generative sampling anywhere. 0 deviations across 11,800+ checks (see §6).

---

## 4. Head-to-head competitor comparison (SummEval, 396 samples)

All tools evaluated on the **identical 396** (doc_id, summary_idx) pairs against the expert
human **consistency** annotation. Same hardware, same human annotations, same evaluation window.

| Tool | Type | Spearman ρ | Pearson r | p | Latency/sample | Cost/sample | Deterministic | n |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **scroot (MiniCheck)** | LLM-free NLI | **0.47** | **0.52** | <0.001 | ~12.4s* | **$0.00** | **Yes** | 396 |
| **scroot (deberta)** | LLM-free NLI | **0.43** | **0.40** | <0.001 | ~5.1s* | **$0.00** | **Yes** | 396 |
| RAGAS faithfulness | LLM judge (gpt-4o-mini) | 0.64 | 0.73 | <0.001 | ~390ms† | $0.00052 | No | 396 |
| DeepEval faithfulness | LLM judge (gpt-4o-mini) | 0.28 | 0.24 | <0.001 | ~8,002ms | $0.00004 | No | 396 |
| TruthScore | (excluded — see note) | — | — | — | — | — | — | — |

\* Groundedness-harness latency (Exp A, groundedness dimension only). Full-pipeline means
(all 5 IQS dimensions): deberta 4,810ms, MiniCheck 8,422ms.

† RAGAS latency is wall-clock/N over a batched parallel `evaluate()` call — not a serial
per-sample figure. It understates RAGAS's true per-request cost; treat cross-tool latency as
indicative.

**scroot (MiniCheck) rho 0.47** and **scroot (deberta) rho 0.43** are from the Exp A
backbone harness, which scores groundedness only on the same 396 samples. The full-pipeline
baseline (all 5 IQS dimensions, deberta backbone) is **rho 0.40** — slightly lower because
inapplicable dimensions (relevance on a generic "Summarize…" query) pull the IQS composite
down. The comparison in this table is groundedness-only vs groundedness-only for all tools.

**Honest framing:**
- RAGAS achieves the highest human correlation (ρ=0.64) — it's the best available signal. But
  it is non-deterministic, API-dependent, and costs ~$0.21 per 396 samples. A pinned LLM still
  varies with server batching; reproducible audits are impossible.
- scroot beats DeepEval (0.47 vs 0.28 with MiniCheck; 0.43 vs 0.28 with deberta) and
  reaches ~73% of RAGAS's correlation — while being the only deterministic, free, offline
  option.
- For production monitoring at scale — where you score every request, need reproducible scores
  for audits, and cannot pay per call — scroot's tradeoff wins. For one-off research evaluation
  where correlation is everything, an LLM judge correlates higher.

**TruthScore excluded:** the `truthscore` PyPI package (v0.3.0) is a reimplementation of
RAGAS FactualCorrectness: it decomposes responses with an LLM, depends on `ragas`, and is
neither LLM-free nor statistically independent of the RAGAS row. Including it would
double-count the RAGAS construct. The genuinely LLM-free NLI faithfulness scorers it could
be confused with (AlignScore, MiniCheck) are not distributed on PyPI. Full exclusion
reasoning: `benchmarks/results/truthscore_exclusion.md`.

---

## 5. Hallucination discrimination (NQ-500, 2,500 records)

scroot's strongest result. Probability that a grounded response scores above a fabricated one:
**AUC 0.991** with MiniCheck — near-perfect hallucination discrimination, deterministic, $0.

### Metric table

| Metric | scroot (MiniCheck, default) | scroot (deberta, fast) | Gate |
|:---|:---:|:---:|:---:|
| **AUC (A0 vs A4)** | **0.991** | 0.875 | ≥ 0.85 **PASS** |
| AUC (A0 vs A3) | 0.967 | 0.968 | — |
| Binary accuracy (threshold 0.5) | 0.991 | 0.876 | — |
| Spearman ρ (groundedness vs level) | -0.863 | -0.693 | — |
| Kendall τ | -0.782 | -0.640 | — |
| Mean separation (A0 − A4) | 0.979 | 0.746 | — |
| Determinism deviations | 0 | 0 | required |

Both backbones clear the ≥ 0.85 gate. MiniCheck improves the deberta AUC from 0.875 → 0.991.

### Per-level gradient (mean groundedness score)

| Level | Description | scroot (MiniCheck) | scroot (deberta) | Monotone? |
|:---:|:---|:---:|:---:|:---:|
| A0 | Correct, grounded | 0.979 | 0.986 | — |
| A1 | Hedged | 0.673 | 0.681 | ↓ ✓ |
| A2 | Mixed grounded+fabricated | 0.586 | 0.600 | ↓ ✓ |
| A3 | Fabricated | 0.048 | 0.061 | ↓ ✓ |
| A4 | Off-topic | 0.000 | 0.240 ⚠ | ↓ ✓ / ↑ ⚠ |

With MiniCheck the gradient is cleanly monotone across all five levels. Deberta has a minor
A4 > A3 inversion (A4 mean 0.240 vs A3 0.061): the IQS harmonic mean already floors both near
zero (A4 IQS ~0.004) so discrimination is not harmed in practice, but MiniCheck resolves the
inversion entirely.

"scroot correctly ranks a grounded response above a fabricated one 99.1% of the time
(AUC 0.991) — near-perfect hallucination discrimination, deterministic and free. This is the
property that matters most for production grounding."

Reproduce: `python benchmarks/bench_minicheck_nq500_gate.py`
Artifact: `benchmarks/results/minicheck_nq500_gate.md`

---

## 6. Determinism

Same input → identical output, bit-for-bit, across all runs. End-to-end through the full
`Auditor.score()` path — not a metric in isolation.

| Check | Examples | Passes | Metrics | Total checks | Deviations |
|:---|:---:|:---:|:---:|:---:|:---:|
| SummEval full pipeline (paper_comparison_final) | 100 | 10 | 6 | 6,000 | **0** |
| top-k premise filtering validation | 100 | 10 | 6 | 6,000 | **0** |
| Composite gating validation | 100 | 10 | 1 | 1,000 | **0** |
| MiniCheck full-pipeline (bench_minicheck_fullpipeline) | 20 | 10 | 1 | 200 | **0** |
| NQ-500 gate determinism (both backbones) | 10 | 10 | 1 | 200 | **0** |
| **Total** | | | | **13,400+** | **0** |

**Why it matters:** reproducible scores for CI gates, compliance audits, and regression
detection. An LLM judge gives different scores for the same input (temperature, server
batching) — it cannot be used where reproducibility is required. scroot can be used as a
deterministic quality gate in GitHub Actions, where a non-zero deviation would be a
build failure.

---

## 7. Performance & latency

Hardware: Intel i7 CPU, single thread, warm cache (models pre-loaded), Windows 11.

### Full-pipeline latency (Auditor.score() — all 5 IQS dimensions)

| Backbone | Mean | p50 | p95 | n | Slowdown vs deberta |
|:---|:---:|:---:|:---:|:---:|:---:|
| deberta-base (default, fast) | 4,810ms | 3,228ms | 14,598ms | 380 | — |
| MiniCheck-RoBERTa-Large (high-accuracy) | 8,422ms | 4,775ms | 28,783ms | 380 | **1.75×** |

**Latency by input type (full pipeline):**

| Input type | deberta mean | MiniCheck mean | Slowdown |
|:---|:---:|:---:|:---:|
| Short RAG, grounded | 3,135ms | 5,296ms | 1.69× |
| Short RAG, hallucinated | 2,381ms | 3,338ms | 1.40× |
| Long document (SummEval) | 12,172ms | 24,003ms | 1.97× |
| No-context fallback | 509ms | 503ms | 0.99× |
| NQ level 1 | 4,927ms | 8,137ms | 1.65× |
| NQ level 2 | 4,311ms | 6,988ms | 1.62× |
| NQ level 3 | 3,369ms | 5,409ms | 1.61× |

Tail latency (p95) is driven by long-document contexts — NLI cost scales with context length.
The p50 is the typical experience; p95 is the long-document worst case. Use focused context
chunks (top-k retrieval) to keep latency in the p50 range.

### Top-k premise pre-filtering (the optimization that cuts long-context cost)

`top_k_premises=8` (default): pre-ranks NLI premises by embedding similarity to the claim and
runs the cross-encoder only on the top-8. Speedup grows with context size; score delta is
0.000 (lossless — the top-k premises are the entailing ones):

| Context sentences | Unfiltered | Filtered (k=5) | Speedup | Score delta |
|:---:|:---:|:---:|:---:|:---:|
| 5 | 1,159ms | 1,587ms | 0.7× | 0.000 |
| 10 | 2,473ms | 1,782ms | 1.4× | 0.000 |
| 20 | 4,200ms | 1,861ms | 2.3× | 0.000 |
| 40 | 7,866ms | 2,220ms | **3.5×** | 0.000 |

100 examples × 10 runs × 6 metrics = 6,000 determinism checks, 0 deviations, across
k ∈ {3, 5, 8, 10}. Reproduce: `python benchmarks/bench_groundedness_topk_accuracy.py`

### Cost at scale

| Volume | scroot | DeepEval (gpt-4o-mini) | RAGAS (gpt-4o-mini) |
|:---|:---:|:---:|:---:|
| 1,000 evals | **$0.00** | ~$0.04 | ~$0.52 |
| 100,000 evals | **$0.00** | ~$4.00 | ~$52.00 |
| 1,000,000 evals/month | **$0.00** | ~$40.00 | ~$520.00 |

DeepEval and RAGAS costs are from the SummEval study: DeepEval $0.00004/sample, RAGAS
$0.00052/sample (gpt-4o-mini, 396 samples). scroot has zero marginal cost at any scale.

---

## 8. What we tried that didn't work (negative results)

These are real hypotheses we tested and rejected on the evidence. Reporting them is a
credibility asset: it shows the numbers came from rigorous experimentation, not cherry-picking.

### spaCy atomic claim decomposition

**Hypothesis:** spaCy dependency-parse decomposition (coordinating conjunctions, relative
clauses, appositives) would extract more precise atomic claims than scroot's regex extractor,
improving SummEval correlation — analogous to what RAGAS does with an LLM.

**Result:** No improvement. On the same 396 SummEval samples (deberta-base backbone,
coverage-ratio aggregation):

| Claim method | Spearman ρ | 95% CI | Pearson r | Latency/sample | Claims/response |
|:---|:---:|:---:|:---:|:---:|:---:|
| regex atomic (current scroot) | 0.4251 | [0.324, 0.520] | 0.4037 | 5.71s | 3.24 |
| spaCy dependency atomic | 0.4160 | [0.310, 0.510] | 0.4040 | 6.75s | 4.12 |

CIs fully overlap. spaCy extracts ~27% more claims and adds ~1s/sample but does not improve
correlation. **Keep the current regex extraction as default.**

Reproduce: `python benchmarks/bench_gap_claim_decomp.py`
Artifact: `benchmarks/results/claim_decomposition.md`

### Alternative aggregation formulas

**Hypothesis:** Mean or minimum support probability would aggregate per-claim NLI scores
better than the current coverage-ratio threshold.

**Result:** Coverage-ratio wins on all three backbones. No change needed:

| Aggregation | deberta ρ | MiniCheck-RoBERTa ρ | MiniCheck-Flan-T5 ρ |
|:---|:---:|:---:|:---:|
| coverage ratio ≥ 0.5 (current) | **0.4251** | **0.4659** | **0.4764** |
| mean support prob | 0.3867 | 0.4629 | 0.4709 |
| min support prob | 0.3895 | 0.4511 | 0.4763 |

Reproduce: `python benchmarks/bench_gap_aggregation.py`
Artifact: `benchmarks/results/aggregation_comparison.md`

### deberta-large vs deberta-base

**Hypothesis:** The larger NLI model would give meaningfully higher SummEval correlation and
justify the latency cost.

**Result:** An early 60-sample run suggested +0.09 ρ improvement. The powered 300-sample A/B
with bootstrap CIs shows the gap collapses into noise:

| NLI backbone | Spearman ρ | 95% CI | Latency/sample |
|:---|:---:|:---:|:---:|
| nli-deberta-v3-base (default) | 0.316 | [0.186, 0.429] | 2,651ms |
| nli-deberta-v3-large | 0.332 | [0.217, 0.437] | 8,783ms |

Paired rho difference (large − base): **+0.016, 95% CI [−0.053, +0.101] — includes zero.**
CIs overlap. The large model costs 3.3× the latency for no robust improvement. **Keep
deberta-base as the default.**

Reproduce: `python benchmarks/bench_model_ab_powered.py`
Artifact: `benchmarks/results/model_ab_powered.md`

### The structural gap to RAGAS

**Finding:** We closed 31.3% of the 0.40→0.64 correlation gap by switching to MiniCheck
(deberta ρ=0.4017 → MiniCheck-RoBERTa ρ=0.4659, MiniCheck-Flan-T5 ρ=0.4764). The
remaining ~0.16 gap appears structural:

- RAGAS uses an LLM judge for claim decomposition — it captures nuance that NLI cross-encoders
  miss (implicit implications, pragmatics, world knowledge).
- The SummEval task (headline-style summaries of news articles) may not be the ideal benchmark
  for scroot's formulation, which is optimised for RAG question-answering.
- Better claim decomposition (spaCy) and aggregation (mean, min) did not help.

Closing the remaining gap requires LLM judges, very large models (7B+), or non-determinism —
all of which conflict with scroot's design goals. We chose not to pursue it further.

---

## 9. Benchmarking history — bugs found and fixed

These bugs were found by benchmarking scroot rigorously against ground truth. Each is
documented with before/after numbers. A scorer you can trust is one whose authors looked hard
enough to find its flaws.

### Bug 1 — NLI paragraph-premise collapse (primary groundedness bug)

**Found in:** A0 investigation (correct NQ answers scoring groundedness = 0.000)

**Symptom:** 8 of 20 A0 responses scored groundedness = 0.000 despite being verbatim
substrings of the context. Example: a 25-word coastal plains sentence extracted directly from
its Wikipedia paragraph scored ENTAIL=0.002, NEUTRAL=0.997 against the full 85-word paragraph.

**Root cause:** `cross-encoder/nli-deberta-v3-base` is trained on sentence pairs where
premise and hypothesis are both single sentences. When the premise is a full paragraph
(50–200 words), the model cannot confidently assign ENTAILMENT and defaults to NEUTRAL. This
is a known NLI cross-encoder training-distribution limitation.

**Fix:** `src/scroot/metrics/groundedness.py` — sentence-split each context chunk before
building NLI pairs. Instead of `(paragraph, claim)`, now runs `(sentence_1, claim),
(sentence_2, claim), …` and takes the best score. This matches the model's training
distribution.

**Effect:**

| | Before fix | After fix |
|:---|:---:|:---:|
| A0 mean groundedness | 0.461 | 0.983 |
| A0 mean IQS | 0.284 | 0.645 |
| NQ-500 AUC (A0 vs A4) | ~0.75 (pre-fix) | 0.875 (deberta) / 0.991 (MiniCheck) |

Artifact: `benchmarks/results/a0_investigation.md`

### Bug 2 — Confidence "May" false-match

**Found in:** A0 investigation (Deadpool 2 release date response scoring confidence = 0.000)

**Symptom:** The response "Deadpool 2 is scheduled to be released on May 18, 2018" scored
confidence = 0.000, collapsing IQS to 0.000.

**Root cause:** `score_confidence` lowercases the response then applies `r'\bmay\b'` as a
hedge pattern. "May 18" → "may 18" → matched as the epistemic hedge "may", giving
hedge_count=1, assert_count=0, score=0.0.

**Fix:** `src/scroot/metrics/confidence.py` — changed pattern to `r'\bmay\b(?!\s*\d)'`
(negative lookahead). "may 18" no longer matches; modal "may" before a verb still matches.

**Effect:** Confidence for the Deadpool response: 0.000 → 0.500 (neutral). IQS: 0.000 → 0.946.

### Bug 3 — Composite collapse on inapplicable dimensions

**Found in:** SummEval IQS correlation investigation (IQS ρ = 0.12 despite groundedness ρ = 0.36)

**Symptom:** On SummEval, IQS had dramatically lower correlation with human annotations than
groundedness alone — despite groundedness being the only dimension with meaningful signal on
this task.

**Root cause:** With a generic "Summarize the following article" query, the relevance dimension
measures query-response cosine similarity — near-constant across all summaries (they all cover
the same topic). This near-zero relevance score collapsed the harmonic mean, pulling IQS far
below the groundedness signal.

**Fix:** `Auditor(gate_inapplicable_dimensions=True)` — detects dimensions that are
structurally inapplicable to a task (via `src/scroot/applicability.py`) and excludes them from
the harmonic mean, renormalising weights over the remaining applicable dimensions. Groundedness
is never gated.

**Effect:**

| | Spearman ρ | n | p |
|:---|:---:|:---:|:---:|
| Ungated IQS (default) | 0.117 | 1,600 | 2.6e-06 |
| **Gated IQS** | **0.248** | 1,600 | 6.7e-24 |

NQ-500 AUC preservation: 0.8651 → 0.8625 (within rounding; gate requirement ≥ 0.85 still
PASS). Groundedness and consistency are never gated; only relevance was gated on SummEval.

The fix is opt-in: `Auditor()` (default) is unchanged. Only
`Auditor(gate_inapplicable_dimensions=True)` activates gating.

Artifact: `benchmarks/results/composite_fix_validation.md`

---

## 10. Reproducibility

### Install

```bash
pip install "scroot[bench]"
```

### Reproduce each result

```bash
# Head-to-head competitor comparison (SummEval 396 samples — needs OPENAI_API_KEY for RAGAS/DeepEval)
python benchmarks/bench_paper_table.py

# Backbone A/B: deberta vs MiniCheck (SummEval, gap-closing experiments)
python benchmarks/bench_gap_backbone_ab.py

# Claim decomposition comparison (Exp B)
python benchmarks/bench_gap_claim_decomp.py

# Aggregation formula comparison (Exp C — uses cached per-claim scores from Exp A)
python benchmarks/bench_gap_aggregation.py

# Combined winner + gap measurement (Exp D)
python benchmarks/bench_gap_final.py --backbone minicheck-flan-t5-large --aggregation coverage

# Powered NLI backbone A/B (300 stratified SummEval samples, 95% CIs)
python benchmarks/bench_model_ab_powered.py

# NQ-500 hallucination discrimination gate
python benchmarks/bench_minicheck_nq500_gate.py

# Full-pipeline integration + latency (Auditor.score(), both backbones, n=380)
python benchmarks/bench_minicheck_fullpipeline.py

# Top-k premise filtering accuracy and speedup
python benchmarks/bench_groundedness_topk_accuracy.py
python benchmarks/bench_groundedness_latency.py

# Composite gating fix validation
python benchmarks/bench_composite_fix_validate.py
```

> **Note on RAGAS/DeepEval reproduction:** these need an `OPENAI_API_KEY` and incur cost
> (~$0.21 for RAGAS, ~$0.02 for DeepEval on 396 samples). RAGAS must be run in an isolated
> venv (`ragas==0.4.3 + langchain 0.2.x`) due to a langchain 1.x incompatibility.
> scroot's numbers are free to reproduce.

### Datasets

- **SummEval:** `datasets` library (`HuggingFaceH4/summarization_human_eval` or
  `mteb/summeval`). All benchmark scripts load it automatically via HuggingFace Hub.
- **NQ-500:** Generated locally from Google Natural Questions:
  ```bash
  python benchmarks/datasets/generate_nq.py
  python benchmarks/datasets/generate_perturbations.py
  ```

### Environment

| | |
|:---|:---|
| CPU | Intel Core i7 (single thread) |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.11 |
| scroot | 0.3.1 |
| sentence-transformers | 4.x |
| transformers | 4.x |
| NLI model (fast) | cross-encoder/nli-deberta-v3-base (184M) |
| NLI model (high-accuracy) | lytang/MiniCheck-RoBERTa-Large (355M) |
| Embedding model | all-MiniLM-L6-v2 (90M) |

### Branch and commit provenance

All numbers in this document trace to committed benchmark code. The results span several
feature branches — not yet merged to main as of this writing:

| Sprint | Branch | Key commit | Numbers produced |
|:---|:---|:---|:---|
| Comparison tightening | `bench/comparison-tightening` | `480a321` | Same-sample 396, RAGAS, DeepEval, powered A/B |
| Gap-closing (Exp A–D) | `bench/gap-closing` | `9cc4da0` | MiniCheck backbones, spaCy, aggregation, gap % |
| NQ-500 gate | `bench/minicheck-nq500-gate` | `dcb2bb7` | AUC 0.991, per-level gradient |
| Full-pipeline integration | `bench/minicheck-fullpipeline` | `0d670c0` | Integrity, determinism, latency n=380 |

**Post-merge re-run needed:** after these branches are merged to main, the SummEval competitor
table (§4) and full-pipeline latency table (§7) should be re-run on main to confirm they
reproduce identically. The NQ-500 discrimination numbers (§5) and determinism checks (§6) are
architecture-stable and do not depend on branch state.

---

## 11. Citation

```bibtex
@software{scroot2026,
  title  = {scroot: LLM-free Response Quality Scoring},
  author = {Guntuka, Sunny},
  year   = {2026},
  url    = {https://github.com/sunnyguntuka/scroot},
  note   = {Apache-2.0}
}
```

A preprint citation will be added here once the arXiv paper is submitted.

---

*scroot v0.3.1 · Apache-2.0 · [github.com/sunnyguntuka/scroot](https://github.com/sunnyguntuka/scroot)*
