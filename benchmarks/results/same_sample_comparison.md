# Same-Sample Comparison: scroot vs DeepEval (396 matched samples)

Both tools evaluated against the human `consistency` annotation (the faithfulness dimension) on the **identical** 396 (doc_id, summary_idx) pairs that DeepEval successfully scored. scroot is NOT re-scored -- its per-sample groundedness scores are read from `summeval_results.json` and filtered to the matched set.

- Matched samples: **396**
- DeepEval excluded 4 of 400 to timeouts; scroot covers all 396 matched.
- human_consistency cross-source mismatches: 0 (0 expected -- same annotation).

## Correlation vs human consistency (same 396 samples)

| Tool | Spearman rho | p | Pearson r | p |
|------|-------------|---|-----------|---|
| **scroot groundedness** (LLM-free) | **0.4017** | <0.001 | 0.3901 | <0.001 |
| DeepEval faithfulness (gpt-4o-mini) | 0.2769 | <0.001 | 0.2393 | <0.001 |

**Headline:** on the identical 396 samples, scroot groundedness Spearman rho = 0.4017 vs DeepEval 0.2769 (delta = +0.1248, scroot higher).

For reference, scroot's full-set rho on all 1,600 SummEval samples is 0.3594; the sprint's earlier 0.36-vs-0.28 comparison mixed sample sizes (1,600 vs 396). The number above is the corrected apples-to-apples figure.
