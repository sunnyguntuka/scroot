"""
Task 5 - Paper-grade consolidated comparison table.

Assembles the tightened results from Tasks 1-4 into a single paper-ready doc:
  benchmarks/results/paper_comparison_final.md

Reads (no scoring here):
  same_sample_comparison.json   (Task 1: scroot + DeepEval on 396)
  ragas_matched.json            (Task 2: RAGAS on the same 396)
  truthscore_exclusion.json     (Task 3: documented exclusion)
  model_ab_powered.json         (Task 4: base vs large, 300, 95% CIs)
  summeval_competitors.json     (DeepEval latency/cost provenance)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)

RESULTS_DIR = Path(__file__).parent / "results"
OUT = RESULTS_DIR / "paper_comparison_final.md"


def _load(name):
    return json.load(open(RESULTS_DIR / name, encoding="utf-8"))


def fmt_p(p):
    if p is None:
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.2g}"


def main():
    ss = _load("same_sample_comparison.json")
    rg = _load("ragas_matched.json")
    ts = _load("truthscore_exclusion.json")
    ab = _load("model_ab_powered.json")
    comp = _load("summeval_competitors.json")

    n = ss["n_matched"]
    sc = ss["scroot_groundedness"]
    de = ss["deepeval_faithfulness_gpt4o_mini"]
    de_meta = comp["summary"]["deepeval"]

    sc_lat = comp["scroot"]["mean_latency_ms"]
    de_lat = de_meta["mean_latency_ms"]
    de_cost = de_meta["cost_usd"]
    de_cost_ps = de_meta["cost_per_sample_usd"]
    rg_lat = rg.get("mean_latency_ms")
    rg_cost = rg["cost_usd"]
    rg_cost_ps = rg_cost / max(len(rg["per_sample"]), 1)
    rg_rho = rg["spearman"]["rho"]
    rg_rho_p = rg["spearman"]["p"]
    rg_r = rg["pearson"]["r"]
    rg_r_p = rg["pearson"]["p"]
    rg_n = len(rg["per_sample"])

    a = ab["model_a"]
    b = ab["model_b"]
    diff = ab["rho_difference_b_minus_a"]

    L = []
    L.append("# Paper-Grade Comparison: scroot vs LLM-judge faithfulness scorers")
    L.append("")
    L.append(
        "All tools evaluated against the human **consistency** annotation "
        "from SummEval (Fabbri et al. 2021) -- the faithfulness dimension -- "
        f"on the **identical {n} (doc_id, summary_idx) pairs**. scroot is not "
        "re-scored; its cached per-sample groundedness is filtered to the "
        "matched set. DeepEval and RAGAS use a gpt-4o-mini judge."
    )
    L.append("")

    L.append(f"## Table 1 - Faithfulness vs human consistency (same {n} samples)")
    L.append("")
    L.append("| Tool | Type | Spearman rho | p | Pearson r | p | n | Determ. | Latency/sample | Cost/sample |")
    L.append("|:-----|:-----|:-----------:|:--:|:--------:|:--:|:-:|:-------:|:--------------:|:-----------:|")
    L.append(
        f"| **scroot groundedness** | LLM-free NLI | **{sc['spearman_rho']:.4f}** | "
        f"{fmt_p(sc['spearman_p'])} | {sc['pearson_r']:.4f} | "
        f"{fmt_p(sc['pearson_p'])} | {n} | **Yes (100%)** | "
        f"{sc_lat:.0f} ms | $0.00 |"
    )
    L.append(
        f"| RAGAS faithfulness | LLM judge (gpt-4o-mini) | {rg_rho:.4f} | "
        f"{fmt_p(rg_rho_p)} | {rg_r:.4f} | {fmt_p(rg_r_p)} | {rg_n} | No | "
        f"{rg_lat:.0f} ms | ${rg_cost_ps:.5f} |"
    )
    L.append(
        f"| DeepEval faithfulness | LLM judge (gpt-4o-mini) | {de['spearman_rho']:.4f} | "
        f"{fmt_p(de['spearman_p'])} | {de['pearson_r']:.4f} | "
        f"{fmt_p(de['pearson_p'])} | {n} | No | {de_lat:.0f} ms | ${de_cost_ps:.5f} |"
    )
    L.append(
        f"| TruthScore | (excluded) | - | - | - | - | - | - | - | - |"
    )
    L.append("")
    L.append(
        f"- **scroot is the only LLM-free, deterministic, zero-cost scorer.** On "
        f"these identical {n} samples it scores rho = {sc['spearman_rho']:.4f}, "
        f"clearly above DeepEval ({de['spearman_rho']:.4f}) and below RAGAS "
        f"({rg_rho:.4f})."
    )
    L.append(
        f"- **RAGAS leads on rank correlation** (rho = {rg_rho:.4f}) but at "
        f"${rg_cost:.2f} total / ${rg_cost_ps*1000:.2f} per 1k samples, "
        f"non-deterministic, and API-dependent. Its faithfulness metric does "
        f"multi-call claim decomposition + NLI per claim, which both costs more "
        f"and tracks human consistency better than DeepEval's single-shot judge."
    )
    L.append(
        f"- **DeepEval** (single FaithfulnessMetric call) trails scroot despite "
        f"using a hosted LLM."
    )
    L.append(
        f"- *Latency note:* RAGAS latency/sample ({rg_lat:.0f} ms) is wall-clock "
        f"time divided by N for a batched, internally-parallel `evaluate()` "
        f"call -- not a serial per-sample figure like scroot's "
        f"({sc_lat:.0f} ms) or DeepEval's ({de_lat:.0f} ms). It understates "
        f"RAGAS's true per-request cost; treat cross-tool latency as indicative."
    )
    L.append(
        f"- **TruthScore excluded:** the `truthscore` PyPI package "
        f"(v{ts['latest_version']}) is a reimplementation of RAGAS "
        f"FactualCorrectness -- LLM-driven (not LLM-free) and dependent on "
        f"`ragas`, so it is redundant with and not independent of the RAGAS row. "
        f"See `truthscore_exclusion.md`."
    )
    L.append("")

    L.append("## Table 2 - Powered NLI backbone A/B (separate 300-sample study)")
    L.append("")
    L.append(
        f"scroot groundedness with two NLI backbones, "
        f"{ab['n_samples']} stratified SummEval samples "
        f"({ab['stratification']}), top_k_premises={ab['top_k_premises']}. "
        f"95% bootstrap CIs from {ab['bootstrap']['n_iter']} paired resamples."
    )
    L.append("")
    L.append("| NLI backbone | Spearman rho | 95% CI | Pearson r | Latency/sample |")
    L.append("|:-------------|:-----------:|:------:|:--------:|:--------------:|")
    L.append(
        f"| nli-deberta-v3-base (default) | {a['spearman_rho']:.4f} | "
        f"[{a['spearman_rho_ci95'][0]:.3f}, {a['spearman_rho_ci95'][1]:.3f}] | "
        f"{a['pearson_r']:.4f} | {a['mean_latency_ms']:.0f} ms |"
    )
    L.append(
        f"| nli-deberta-v3-large | {b['spearman_rho']:.4f} | "
        f"[{b['spearman_rho_ci95'][0]:.3f}, {b['spearman_rho_ci95'][1]:.3f}] | "
        f"{b['pearson_r']:.4f} | {b['mean_latency_ms']:.0f} ms |"
    )
    L.append("")
    L.append(
        f"Paired rho difference (large - base) = {diff['point']:+.4f}, "
        f"95% CI [{diff['ci95'][0]:.3f}, {diff['ci95'][1]:.3f}] "
        f"({'excludes' if diff['excludes_zero'] else 'includes'} zero). "
        f"Per-model CIs {'overlap' if ab['per_model_ci_overlap'] else 'are disjoint'}."
    )
    L.append("")
    L.append(f"**{ab['conclusion']}**")
    L.append("")

    L.append("## Methodology")
    L.append("")
    L.append(
        f"- **Dataset / annotations:** SummEval (Fabbri et al. 2021), "
        f"100 CNN/DailyMail articles x 16 system summaries = 1,600 samples with "
        f"expert human annotations (consistency, relevance, coherence, fluency). "
        f"The faithfulness target is the mean expert **consistency** score."
    )
    L.append(
        f"- **Sample selection (Table 1):** the {n} samples are exactly those "
        f"DeepEval successfully scored in the prior sprint (4 of 400 stratified "
        f"samples failed to gpt-4o-mini timeouts and were excluded for all "
        f"tools). scroot and RAGAS were aligned to that same set by "
        f"(doc_id, summary_idx); 0 human-consistency mismatches across sources."
    )
    L.append(
        f"- **Sample selection (Table 2):** {ab['n_samples']} samples, "
        f"60 from each of 5 human-consistency RANK tiers (consistency is "
        f"skewed toward 5.0, so value quantiles collapse; rank bands guarantee "
        f"tail coverage). Deterministic ordering by "
        f"(consistency, doc_id, summary_idx)."
    )
    L.append(
        f"- **Judge model:** gpt-4o-mini (temperature 0) for DeepEval and RAGAS. "
        f"RAGAS via isolated venv ragas=={rg['ragas_version']} + langchain 0.2.x "
        f"(the main env's langchain 1.x is incompatible with ragas 0.4.3)."
    )
    L.append(
        f"- **scroot config:** groundedness dimension, default "
        f"nli-deberta-v3-base backbone, all-MiniLM-L6-v2 embeddings, "
        f"top_k_premises=8 (lossless: MAD=0.0 vs uncapped). Fully deterministic "
        f"(0/5400 deviations over 10 runs)."
    )
    L.append(
        f"- **Statistics:** Spearman rho and Pearson r with two-sided p-values "
        f"(scipy). Table 2 adds 95% percentile bootstrap CIs "
        f"({ab['bootstrap']['n_iter']} paired resamples, seed "
        f"{ab['bootstrap']['seed']})."
    )
    L.append(
        f"- **Hardware:** local CPU (Windows 11). scroot latencies are CPU "
        f"NLI+embedding inference; LLM-judge latencies are dominated by OpenAI "
        f"API round-trips."
    )
    L.append(
        f"- **Cost:** scroot $0.00 (local). RAGAS ${rg_cost:.4f} total "
        f"(${rg_cost_ps:.5f}/sample). DeepEval ${de_cost:.4f} total "
        f"(${de_cost_ps:.5f}/sample). Total API spend for this study: "
        f"${rg_cost + de_cost:.4f} (DeepEval cost was incurred in the prior "
        f"sprint; this sprint added only the ${rg_cost:.4f} RAGAS run)."
    )
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(
        "Tables regenerate from `same_sample_comparison.json`, "
        "`ragas_matched.json`, `model_ab_powered.json`, "
        "`truthscore_exclusion.json`, `summeval_competitors.json` via "
        "`python benchmarks/bench_paper_table.py`."
    )
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Table 1: scroot {sc['spearman_rho']:.4f} | RAGAS {rg_rho:.4f} | "
          f"DeepEval {de['spearman_rho']:.4f} (n={n})")
    print(f"Table 2: base {a['spearman_rho']:.4f} "
          f"{a['spearman_rho_ci95']} vs large {b['spearman_rho']:.4f} "
          f"{b['spearman_rho_ci95']}; diff CI {diff['ci95']} "
          f"excludes_zero={diff['excludes_zero']}")


if __name__ == "__main__":
    main()
