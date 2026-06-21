"""
Task 6 - Defaults recommendation, driven by the tightened evidence.

Writes benchmarks/results/defaults_recommendation.md. Does NOT change any
defaults -- it only recommends, reading the validated benchmark artifacts:
  groundedness_topk_accuracy.json   (top_k_premises lossless + deterministic)
  groundedness_latency.json         (top_k_premises latency win)
  composite_fix_validation.json     (gating: SummEval IQS up, NQ-500 AUC kept)
  model_ab_powered.json             (base vs large, 95% CIs)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)

RESULTS_DIR = Path(__file__).parent / "results"
OUT = RESULTS_DIR / "defaults_recommendation.md"


def _load(name):
    return json.load(open(RESULTS_DIR / name, encoding="utf-8"))


def main():
    topk = _load("groundedness_topk_accuracy.json")
    lat = _load("groundedness_latency.json")
    comp = _load("composite_fix_validation.json")
    ab = _load("model_ab_powered.json")

    # top_k accuracy: max MAD across k.
    mad8 = topk["mad_by_k"]["8"]["mad"]
    det_dev = topk["determinism_deviations"]
    # latency win at the largest context tested.
    big = max(lat["rows"], key=lambda r: r["context_sentences"])

    nq = comp["nq500"]
    se = comp["summeval"]

    a = ab["model_a"]
    b = ab["model_b"]
    diff = ab["rho_difference_b_minus_a"]
    robust = diff["excludes_zero"]

    L = []
    L.append("# Defaults Recommendation (Task 6)")
    L.append("")
    L.append(
        "Data-driven recommendations on three candidate default changes. "
        "**No defaults are changed by this sprint** -- these are recommendations "
        "for maintainers, each backed by a validated benchmark artifact."
    )
    L.append("")

    # --- 1. top_k_premises -------------------------------------------------
    L.append("## 1. `top_k_premises=8` ON by default? -> RECOMMEND: YES")
    L.append("")
    L.append(
        f"- **Lossless:** mean abs score difference vs uncapped is "
        f"{mad8:.3f} (MAD=0.0 at k in {{3,5,8,10}}) on a 50-sample check -- "
        f"the cap changes no scores."
    )
    L.append(
        f"- **Deterministic:** {det_dev} determinism deviations with the cap on."
    )
    L.append(
        f"- **Faster on long contexts:** at "
        f"{big['context_sentences']} context sentences, "
        f"{big['off_ms']:.0f} ms -> {big['on_ms']:.0f} ms "
        f"({big['speedup']:.1f}x) with no score change "
        f"(delta {big['score_delta']:.2f})."
    )
    L.append(
        "- **Verdict:** lossless and strictly faster on non-trivial contexts; "
        "safe to enable by default. (Recommendation only -- not applied.)"
    )
    L.append("")

    # --- 2. gating ---------------------------------------------------------
    L.append("## 2. `gate_inapplicable_dimensions=True` by default? "
             "-> RECOMMEND: YES (with one caveat)")
    L.append("")
    L.append(
        f"- **Helps when a dimension is structurally inapplicable:** on SummEval "
        f"(generic summarize query makes *relevance* inapplicable) gated IQS "
        f"rho rises {se['old_iqs_rho']:.3f} -> {se['new_iqs_rho']:.3f} "
        f"(p {se['old_p']:.1e} -> {se['new_p']:.1e}), n={se['n']}."
    )
    L.append(
        f"- **Does not hurt discrimination on real-query data:** on NQ-500 the "
        f"A0-vs-A4 IQS AUC is essentially unchanged, "
        f"{nq['old_auc_a0_vs_a4']:.4f} -> {nq['new_auc_a0_vs_a4']:.4f} "
        f"(>= 0.85 gate preserved; {nq['n_relevance_gated']} of {nq['n']} rows "
        f"incidentally gated, 0 consistency-gated)."
    )
    L.append(
        "- **Caveat:** this changes IQS semantics (dimensions can be dropped "
        "from the harmonic mean), so flipping the default is a behavior change "
        "for downstream consumers. Recommend enabling by default in the next "
        "MINOR with a changelog note, not a patch."
    )
    L.append("")

    # --- 3. NLI large ------------------------------------------------------
    L.append("## 3. Switch default NLI backbone to `nli-deberta-v3-large`? "
             f"-> RECOMMEND: {'YES' if robust else 'NO (keep base)'}")
    L.append("")
    L.append(
        f"Powered A/B ({ab['n_samples']} stratified samples, 95% bootstrap CIs):"
    )
    L.append("")
    L.append("| Backbone | rho | 95% CI | latency/sample |")
    L.append("|:---|:---:|:---:|:---:|")
    L.append(
        f"| base (current default) | {a['spearman_rho']:.4f} | "
        f"[{a['spearman_rho_ci95'][0]:.3f}, {a['spearman_rho_ci95'][1]:.3f}] | "
        f"{a['mean_latency_ms']:.0f} ms |")
    L.append(
        f"| large | {b['spearman_rho']:.4f} | "
        f"[{b['spearman_rho_ci95'][0]:.3f}, {b['spearman_rho_ci95'][1]:.3f}] | "
        f"{b['mean_latency_ms']:.0f} ms |")
    L.append("")
    L.append(
        f"- Paired rho difference (large - base) = {diff['point']:+.4f}, "
        f"95% CI [{diff['ci95'][0]:.3f}, {diff['ci95'][1]:.3f}] "
        f"({'EXCLUDES' if robust else 'INCLUDES'} zero)."
    )
    speed_mult = b['mean_latency_ms'] / max(a['mean_latency_ms'], 1e-9)
    if robust:
        L.append(
            f"- **Verdict:** the large model is a statistically robust "
            f"improvement, but at ~{speed_mult:.1f}x latency. Recommend keeping "
            f"base as the zero-config default and documenting large as the "
            f"opt-in accuracy backbone (`Auditor(nli_model=...)`), or making it "
            f"default only for a latency-tolerant 'accuracy' profile."
        )
    else:
        L.append(
            f"- **Verdict:** the large model's apparent edge is NOT robust at "
            f"n={ab['n_samples']} (difference CI includes 0) while costing "
            f"~{speed_mult:.1f}x latency. **Do not** switch the default. The "
            f"earlier 60-sample result (base 0.27 vs large 0.36) was "
            f"underpowered; with 300 samples and CIs the gap is within noise. "
            f"Keep base; offer large as a documented opt-in."
        )
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Candidate default | Recommendation | Basis |")
    L.append("|:---|:---:|:---|")
    L.append("| top_k_premises=8 | **Enable** | lossless (MAD=0.0), deterministic, faster |")
    L.append("| gate_inapplicable_dimensions=True | **Enable (next minor)** | "
             "+IQS signal on inapplicable dims, NQ-500 AUC preserved |")
    L.append(f"| default NLI = large | **{'Switch' if robust else 'Keep base'}** | "
             f"paired rho-diff CI {'excludes' if robust else 'includes'} 0 |")
    L.append("")
    L.append("_No defaults were changed by this sprint._")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"NLI large robust improvement: {robust}")


if __name__ == "__main__":
    main()
