"""
Task 1 - Same-sample comparison: scroot vs DeepEval on the IDENTICAL 396
samples DeepEval actually scored.

The yesterday's sprint reported scroot groundedness rho=0.36 on 1,600 SummEval
samples but DeepEval rho=0.28 on only 396 samples -- not apples-to-apples.
This script restricts scroot to the SAME 396 (doc_id, summary_idx) pairs and
recomputes Spearman rho / Pearson r / p-values so the comparison is fair.

No re-scoring: scroot scores are read from summeval_results.json.

Outputs:
  benchmarks/results/same_sample_comparison.md
  benchmarks/results/same_sample_comparison.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# benchmarks/ shadows the HuggingFace `datasets` package -- drop it from path.
_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from scipy import stats  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
COMPETITORS_PATH = RESULTS_DIR / "summeval_competitors.json"
SCROOT_RESULTS_PATH = RESULTS_DIR / "summeval_results.json"
OUT_MD = RESULTS_DIR / "same_sample_comparison.md"
OUT_JSON = RESULTS_DIR / "same_sample_comparison.json"


def key(doc_id, summary_idx):
    return (doc_id, int(summary_idx))


def main():
    comp = json.load(open(COMPETITORS_PATH, encoding="utf-8"))
    res = json.load(open(SCROOT_RESULTS_PATH, encoding="utf-8"))

    deepeval = comp["deepeval_raw"]["per_sample"]
    # The 396 samples DeepEval actually scored (errors already excluded).
    deepeval_keys = [key(d["doc_id"], d["summary_idx"]) for d in deepeval]
    print(f"DeepEval scored {len(deepeval_keys)} samples")

    # Index scroot scores by (doc_id, summary_idx).
    scroot_by_key = {
        key(s["doc_id"], s["summary_idx"]): s for s in res["per_sample_scores"]
    }

    # Build aligned arrays on the matched samples.
    scroot_g, scroot_human = [], []
    deepeval_s, deepeval_human = [], []
    matched = 0
    missing = []
    for d in deepeval:
        k = key(d["doc_id"], d["summary_idx"])
        s = scroot_by_key.get(k)
        if s is None:
            missing.append(k)
            continue
        matched += 1
        scroot_g.append(s["scroot_groundedness"])
        scroot_human.append(s["human_consistency"])
        deepeval_s.append(d["score"])
        deepeval_human.append(d["human_consistency"])

    print(f"Matched {matched} samples; missing in scroot results: {len(missing)}")
    if missing:
        print("  e.g.", missing[:3])

    # Sanity: human_consistency must agree between the two sources.
    mism = sum(
        1 for a, b in zip(scroot_human, deepeval_human) if abs(a - b) > 1e-6
    )
    print(f"human_consistency mismatches between sources: {mism}")

    # scroot groundedness vs human consistency on the SAME 396.
    sc_rho, sc_rho_p = stats.spearmanr(scroot_g, scroot_human)
    sc_r, sc_r_p = stats.pearsonr(scroot_g, scroot_human)

    # DeepEval vs human consistency (recomputed on the matched set for symmetry).
    de_rho, de_rho_p = stats.spearmanr(deepeval_s, deepeval_human)
    de_r, de_r_p = stats.pearsonr(deepeval_s, deepeval_human)

    n = matched
    out = {
        "task": "same_sample_comparison",
        "n_matched": n,
        "human_consistency_mismatches": mism,
        "scroot_groundedness": {
            "spearman_rho": round(sc_rho, 4),
            "spearman_p": float(f"{sc_rho_p:.3g}"),
            "pearson_r": round(sc_r, 4),
            "pearson_p": float(f"{sc_r_p:.3g}"),
        },
        "deepeval_faithfulness_gpt4o_mini": {
            "spearman_rho": round(de_rho, 4),
            "spearman_p": float(f"{de_rho_p:.3g}"),
            "pearson_r": round(de_r, 4),
            "pearson_p": float(f"{de_r_p:.3g}"),
        },
        "scroot_full_1600_reference": {
            "spearman_rho": res["correlations"][
                "groundedness_vs_human_consistency"
            ]["spearman_rho"],
            "n": 1600,
        },
    }
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=2)

    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3g}"

    lines = []
    lines.append("# Same-Sample Comparison: scroot vs DeepEval (396 matched samples)")
    lines.append("")
    lines.append(
        "Both tools evaluated against the human `consistency` annotation "
        "(the faithfulness dimension) on the **identical** "
        f"{n} (doc_id, summary_idx) pairs that DeepEval successfully scored. "
        "scroot is NOT re-scored -- its per-sample groundedness scores are "
        "read from `summeval_results.json` and filtered to the matched set."
    )
    lines.append("")
    lines.append(f"- Matched samples: **{n}**")
    lines.append(
        f"- DeepEval excluded 4 of 400 to timeouts; scroot covers all "
        f"{n} matched."
    )
    lines.append(
        f"- human_consistency cross-source mismatches: {mism} "
        "(0 expected -- same annotation)."
    )
    lines.append("")
    lines.append("## Correlation vs human consistency (same 396 samples)")
    lines.append("")
    lines.append("| Tool | Spearman rho | p | Pearson r | p |")
    lines.append("|------|-------------|---|-----------|---|")
    lines.append(
        f"| **scroot groundedness** (LLM-free) | **{sc_rho:.4f}** | "
        f"{fmt_p(sc_rho_p)} | {sc_r:.4f} | {fmt_p(sc_r_p)} |"
    )
    lines.append(
        f"| DeepEval faithfulness (gpt-4o-mini) | {de_rho:.4f} | "
        f"{fmt_p(de_rho_p)} | {de_r:.4f} | {fmt_p(de_r_p)} |"
    )
    lines.append("")
    delta = sc_rho - de_rho
    winner = "scroot" if delta > 0 else "DeepEval"
    lines.append(
        f"**Headline:** on the identical {n} samples, scroot groundedness "
        f"Spearman rho = {sc_rho:.4f} vs DeepEval {de_rho:.4f} "
        f"(delta = {delta:+.4f}, {winner} higher)."
    )
    lines.append("")
    lines.append(
        f"For reference, scroot's full-set rho on all 1,600 SummEval samples "
        f"is {out['scroot_full_1600_reference']['spearman_rho']:.4f}; the "
        f"sprint's earlier 0.36-vs-0.28 comparison mixed sample sizes "
        f"(1,600 vs 396). The number above is the corrected apples-to-apples "
        f"figure."
    )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== SAME-SAMPLE RESULT (n=%d) ===" % n)
    print(f"scroot groundedness rho={sc_rho:.4f} p={fmt_p(sc_rho_p)}")
    print(f"DeepEval rho={de_rho:.4f} p={fmt_p(de_rho_p)}")
    print(f"delta={delta:+.4f} -> {winner} higher")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
