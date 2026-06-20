# Paper-Grade Comparison: scroot vs LLM-judge faithfulness scorers

All tools evaluated against the human **consistency** annotation from SummEval (Fabbri et al. 2021) -- the faithfulness dimension -- on the **identical 396 (doc_id, summary_idx) pairs**. scroot is not re-scored; its cached per-sample groundedness is filtered to the matched set. DeepEval and RAGAS use a gpt-4o-mini judge.

## Table 1 - Faithfulness vs human consistency (same 396 samples)

| Tool | Type | Spearman rho | p | Pearson r | p | n | Determ. | Latency/sample | Cost/sample |
|:-----|:-----|:-----------:|:--:|:--------:|:--:|:-:|:-------:|:--------------:|:-----------:|
| **scroot groundedness** | LLM-free NLI | **0.4017** | <0.001 | 0.3901 | <0.001 | 396 | **Yes (100%)** | 8588 ms | $0.00 |
| RAGAS faithfulness | LLM judge (gpt-4o-mini) | 0.6440 | <0.001 | 0.7301 | <0.001 | 396 | No | 390 ms | $0.00052 |
| DeepEval faithfulness | LLM judge (gpt-4o-mini) | 0.2769 | <0.001 | 0.2393 | <0.001 | 396 | No | 8002 ms | $0.00004 |
| TruthScore | (excluded) | - | - | - | - | - | - | - | - |

- **scroot is the only LLM-free, deterministic, zero-cost scorer.** On these identical 396 samples it scores rho = 0.4017, clearly above DeepEval (0.2769) and below RAGAS (0.6440).
- **RAGAS leads on rank correlation** (rho = 0.6440) but at $0.21 total / $0.52 per 1k samples, non-deterministic, and API-dependent. Its faithfulness metric does multi-call claim decomposition + NLI per claim, which both costs more and tracks human consistency better than DeepEval's single-shot judge.
- **DeepEval** (single FaithfulnessMetric call) trails scroot despite using a hosted LLM.
- *Latency note:* RAGAS latency/sample (390 ms) is wall-clock time divided by N for a batched, internally-parallel `evaluate()` call -- not a serial per-sample figure like scroot's (8588 ms) or DeepEval's (8002 ms). It understates RAGAS's true per-request cost; treat cross-tool latency as indicative.
- **TruthScore excluded:** the `truthscore` PyPI package (v0.3.0) is a reimplementation of RAGAS FactualCorrectness -- LLM-driven (not LLM-free) and dependent on `ragas`, so it is redundant with and not independent of the RAGAS row. See `truthscore_exclusion.md`.

## Table 2 - Powered NLI backbone A/B (separate 300-sample study)

scroot groundedness with two NLI backbones, 300 stratified SummEval samples (5 rank tiers x 60/tier), top_k_premises=8. 95% bootstrap CIs from 1000 paired resamples.

| NLI backbone | Spearman rho | 95% CI | Pearson r | Latency/sample |
|:-------------|:-----------:|:------:|:--------:|:--------------:|
| nli-deberta-v3-base (default) | 0.3158 | [0.186, 0.429] | 0.3776 | 2651 ms |
| nli-deberta-v3-large | 0.3318 | [0.217, 0.437] | 0.4100 | 8783 ms |

Paired rho difference (large - base) = +0.0160, 95% CI [-0.053, 0.101] (includes zero). Per-model CIs overlap.

**No robust improvement: the paired rho-difference 95% CI includes 0, so the large model's apparent edge is within sampling noise.**

## Methodology

- **Dataset / annotations:** SummEval (Fabbri et al. 2021), 100 CNN/DailyMail articles x 16 system summaries = 1,600 samples with expert human annotations (consistency, relevance, coherence, fluency). The faithfulness target is the mean expert **consistency** score.
- **Sample selection (Table 1):** the 396 samples are exactly those DeepEval successfully scored in the prior sprint (4 of 400 stratified samples failed to gpt-4o-mini timeouts and were excluded for all tools). scroot and RAGAS were aligned to that same set by (doc_id, summary_idx); 0 human-consistency mismatches across sources.
- **Sample selection (Table 2):** 300 samples, 60 from each of 5 human-consistency RANK tiers (consistency is skewed toward 5.0, so value quantiles collapse; rank bands guarantee tail coverage). Deterministic ordering by (consistency, doc_id, summary_idx).
- **Judge model:** gpt-4o-mini (temperature 0) for DeepEval and RAGAS. RAGAS via isolated venv ragas==0.4.3 + langchain 0.2.x (the main env's langchain 1.x is incompatible with ragas 0.4.3).
- **scroot config:** groundedness dimension, default nli-deberta-v3-base backbone, all-MiniLM-L6-v2 embeddings, top_k_premises=8 (lossless: MAD=0.0 vs uncapped). Fully deterministic (0/5400 deviations over 10 runs).
- **Statistics:** Spearman rho and Pearson r with two-sided p-values (scipy). Table 2 adds 95% percentile bootstrap CIs (1000 paired resamples, seed 1234).
- **Hardware:** local CPU (Windows 11). scroot latencies are CPU NLI+embedding inference; LLM-judge latencies are dominated by OpenAI API round-trips.
- **Cost:** scroot $0.00 (local). RAGAS $0.2051 total ($0.00052/sample). DeepEval $0.0173 total ($0.00004/sample). Total API spend for this study: $0.2224 (DeepEval cost was incurred in the prior sprint; this sprint added only the $0.2051 RAGAS run).

## Provenance

Tables regenerate from `same_sample_comparison.json`, `ragas_matched.json`, `model_ab_powered.json`, `truthscore_exclusion.json`, `summeval_competitors.json` via `python benchmarks/bench_paper_table.py`.
