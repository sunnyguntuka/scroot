# Defaults Recommendation (Task 6)

Data-driven recommendations on three candidate default changes. **No defaults are changed by this sprint** -- these are recommendations for maintainers, each backed by a validated benchmark artifact.

## 1. `top_k_premises=8` ON by default? -> RECOMMEND: YES

- **Lossless:** mean abs score difference vs uncapped is 0.000 (MAD=0.0 at k in {3,5,8,10}) on a 50-sample check -- the cap changes no scores.
- **Deterministic:** 0 determinism deviations with the cap on.
- **Faster on long contexts:** at 40 context sentences, 7866 ms -> 2220 ms (3.5x) with no score change (delta 0.00).
- **Verdict:** lossless and strictly faster on non-trivial contexts; safe to enable by default. (Recommendation only -- not applied.)

## 2. `gate_inapplicable_dimensions=True` by default? -> RECOMMEND: YES (with one caveat)

- **Helps when a dimension is structurally inapplicable:** on SummEval (generic summarize query makes *relevance* inapplicable) gated IQS rho rises 0.117 -> 0.248 (p 2.6e-06 -> 6.7e-24), n=1600.
- **Does not hurt discrimination on real-query data:** on NQ-500 the A0-vs-A4 IQS AUC is essentially unchanged, 0.8651 -> 0.8625 (>= 0.85 gate preserved; 380 of 2500 rows incidentally gated, 0 consistency-gated).
- **Caveat:** this changes IQS semantics (dimensions can be dropped from the harmonic mean), so flipping the default is a behavior change for downstream consumers. Recommend enabling by default in the next MINOR with a changelog note, not a patch.

## 3. Switch default NLI backbone to `nli-deberta-v3-large`? -> RECOMMEND: NO (keep base)

Powered A/B (300 stratified samples, 95% bootstrap CIs):

| Backbone | rho | 95% CI | latency/sample |
|:---|:---:|:---:|:---:|
| base (current default) | 0.3158 | [0.186, 0.429] | 2651 ms |
| large | 0.3318 | [0.217, 0.437] | 8783 ms |

- Paired rho difference (large - base) = +0.0160, 95% CI [-0.053, 0.101] (INCLUDES zero).
- **Verdict:** the large model's apparent edge is NOT robust at n=300 (difference CI includes 0) while costing ~3.3x latency. **Do not** switch the default. The earlier 60-sample result (base 0.27 vs large 0.36) was underpowered; with 300 samples and CIs the gap is within noise. Keep base; offer large as a documented opt-in.

## Summary

| Candidate default | Recommendation | Basis |
|:---|:---:|:---|
| top_k_premises=8 | **Enable** | lossless (MAD=0.0), deterministic, faster |
| gate_inapplicable_dimensions=True | **Enable (next minor)** | +IQS signal on inapplicable dims, NQ-500 AUC preserved |
| default NLI = large | **Keep base** | paired rho-diff CI includes 0 |

_No defaults were changed by this sprint._
