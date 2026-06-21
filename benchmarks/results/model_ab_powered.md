# Powered Model A/B: NLI base vs large (Task 4)

SummEval groundedness vs human `consistency`, **300 stratified samples** (5 rank tiers x 60/tier), top_k_premises=8. 95% bootstrap CIs from 1000 paired resamples (seed 1234).

| Model | Spearman rho | 95% CI | Pearson r | mean latency |
|-------|-------------|--------|-----------|--------------|
| A: nli-deberta-v3-base (default) | 0.3158 | [0.186, 0.429] | 0.3776 | 2651 ms |
| B: nli-deberta-v3-large | 0.3318 | [0.217, 0.437] | 0.4100 | 8783 ms |

**Paired rho difference (B - A):** +0.0160, 95% CI [-0.053, 0.101] -- INCLUDES zero.

**Per-model CI overlap:** YES.

**Conclusion:** No robust improvement: the paired rho-difference 95% CI includes 0, so the large model's apparent edge is within sampling noise.

Inter-model agreement: mean |delta groundedness| = 0.0694, inter-model Spearman = 0.7784.

Note: the paired rho-difference CI is the powered test of record; per-model CIs ignore the fact that both models score the same samples and so overlap even when the paired difference is significant.
