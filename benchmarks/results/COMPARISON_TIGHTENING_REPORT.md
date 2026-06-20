# Comparison Tightening Sprint - Final Report

Branch: `bench/comparison-tightening`. Goal: make scroot's SummEval competitor
comparison paper-grade -- identical samples for all tools, RAGAS included,
TruthScore resolved, model A/B properly powered with confidence intervals.

## 1. Same-sample headline: does scroot still beat DeepEval?

**Yes -- by a larger margin than before.** On the **identical 396**
(doc_id, summary_idx) pairs DeepEval scored (all 396 matched in scroot's cache,
0 human-consistency mismatches across sources):

| Tool | Spearman rho | p | Pearson r |
|:-----|:-----------:|:--:|:--------:|
| **scroot groundedness** (LLM-free) | **0.4017** | <0.001 | 0.3901 |
| DeepEval faithfulness (gpt-4o-mini) | 0.2769 | <0.001 | 0.2393 |

Delta = **+0.1248** in scroot's favour. The prior sprint's headline
(scroot 0.36 on 1,600 vs DeepEval 0.28 on 396) mixed sample sizes; restricted
to the same 396, scroot's groundedness rho is actually **0.40**, so the gap is
wider, not narrower. Artifact: `same_sample_comparison.md`.

## 2. RAGAS: result (now runs)

**Ran successfully**, all 396 samples scored, 0 excluded.

| Tool | Spearman rho | p | Pearson r | Cost |
|:-----|:-----------:|:--:|:--------:|:----:|
| RAGAS faithfulness (gpt-4o-mini) | **0.6440** | <0.001 | 0.7301 | $0.2051 |

**Fix:** ragas 0.4.3 (the latest on PyPI -- `ragas>=0.5` does not exist) imports
`langchain_community.chat_models.vertexai`, a path removed in
langchain-community 0.4.x present in the main env. Resolved with an isolated
venv (`.ragas-env`) pinned to ragas==0.4.3 + langchain 0.2.17 +
langchain-community 0.2.19 + openai 1.109, restoring the import path. Run via
`.ragas-env/Scripts/python.exe benchmarks/bench_ragas_matched.py`.

**Honest finding:** RAGAS faithfulness **outperforms scroot** on this task
(0.64 vs 0.40). Its metric does multi-call LLM claim decomposition + per-claim
NLI, which tracks human consistency well -- but it is non-deterministic,
API-dependent, and costs ~$0.21 per 396 (vs scroot's $0 and 100% determinism).
This belongs in the paper as a fair characterization, not a hidden result.
Artifact: `ragas_matched.md`.

## 3. TruthScore: documented exclusion

A `truthscore` package **does** exist on PyPI (v0.3.0) and **is** installable
(no dependency conflict with the ragas venv). It is **excluded** -- not because
it won't install, but because it does not fit the role:

- It is a **reimplementation of RAGAS's FactualCorrectness** metric, uses an
  **LLM for claim decomposition (not LLM-free)**, and depends on `ragas`.
- It would therefore be **redundant with and not statistically independent of**
  the RAGAS row already reported -- double-counting the same construct.
- The genuinely LLM-free NLI faithfulness scorers the name could mean
  (AlignScore, MiniCheck) have **no PyPI distribution** and were out of scope.

Formal exclusion paragraph in `truthscore_exclusion.md`.

## 4. Powered model A/B: base vs large with 95% CIs

300 stratified samples (5 rank tiers x 60), groundedness-only,
top_k_premises=8, 1000-iteration paired bootstrap (seed 1234):

| NLI backbone | Spearman rho | 95% CI | Latency/sample |
|:-------------|:-----------:|:------:|:--------------:|
| nli-deberta-v3-base (default) | 0.3158 | [0.186, 0.429] | 2,651 ms |
| nli-deberta-v3-large | 0.3318 | [0.217, 0.437] | 8,783 ms |

- Paired rho difference (large - base) = **+0.0160**, 95% CI **[-0.053, 0.101]
  -- includes zero**. Per-model CIs **overlap**.
- **Conclusion: NOT a robust improvement.** The large model's apparent edge is
  within sampling noise at n=300.

This **overturns** the prior 60-sample result (base 0.27 vs large 0.36,
"+0.09 rho, meaningfully better"), which was underpowered. With proper power and
CIs the difference collapses to +0.016 and is not significant. Artifact:
`model_ab_powered.md`.

## 5. Consolidated table

Full paper-grade tables (faithfulness comparison + powered A/B) with type,
determinism, latency, cost, p-values, and a methodology section:
**`paper_comparison_final.md`**.

## 6. Defaults recommendation (no defaults changed)

| Candidate default | Recommendation | Basis |
|:---|:---:|:---|
| `top_k_premises=8` | **Enable** | lossless (MAD=0.0 vs uncapped), deterministic, up to 3.5x faster on long contexts |
| `gate_inapplicable_dimensions=True` | **Enable in next minor** | SummEval IQS rho 0.117 -> 0.248; NQ-500 AUC preserved (0.8651 -> 0.8625, >= 0.85) |
| default NLI = large | **Keep base** | powered A/B difference CI includes 0; large costs ~3.3x latency for no robust gain |

Detail and caveats in `defaults_recommendation.md`. **No defaults were changed.**

## 7. What's paper-ready now vs remaining caveats

**Paper-ready:**
- Identical-396 faithfulness comparison (scroot, RAGAS, DeepEval) with rho, r,
  p-values, determinism, latency, cost.
- RAGAS now reproducibly runs (pinned isolated venv documented).
- Powered, CI-backed model A/B that corrects the underpowered earlier claim.
- Formal TruthScore exclusion.
- Methodology paragraph (dataset, annotation source, sample selection, judge,
  hardware, statistics).

**Remaining caveats:**
- Cross-tool **latency** is only indicative: RAGAS's per-sample figure is
  wall-clock/N over a batched, internally-parallel `evaluate()`, not serial like
  scroot/DeepEval. Cost/determinism comparisons are exact; latency is not.
- DeepEval's 4/400 timeout exclusions define the 396-sample set; all tools were
  aligned to that set for fairness, so n is judge-limited, not scroot-limited.
- RAGAS's higher rho comes at non-determinism + API cost + a heavier multi-call
  pipeline; the paper should frame scroot as the best *LLM-free, deterministic,
  zero-cost* scorer rather than the highest rho overall.
- A/B and Table-2 latencies are local single-CPU, single-run; not multi-trial
  benchmarked.

## 8. Total API cost

- RAGAS run (this sprint): **$0.2051**
- DeepEval (prior sprint, reused, not re-incurred): $0.0173
- Powered model A/B: **$0.00** (local NLI inference)
- **New API spend this sprint: ~$0.21.** Far under the $30 cost guard; guard
  never tripped.

## Commits (branch `bench/comparison-tightening`)

| Task | Commit | Subject |
|:---|:---|:---|
| 1 | 67a8ba9 | same-sample scroot vs DeepEval on 396 matched samples |
| 2 | 7efd38e | RAGAS faithfulness on 396 matched samples |
| 3 | e0773b7 | TruthScore - install attempt and formal exclusion note |
| 4 | 457e47d | powered model A/B (300 samples, 95% CI) |
| 5 | aac5431 | paper-grade comparison table |
| 6 | 50fa82d | defaults recommendation based on tightened evidence |

(Final-report + BENCHMARKS.md commit follows.)
