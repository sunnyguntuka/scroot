"""
Task 4 - POWERED NLI model A/B on SummEval groundedness with bootstrap CIs.

Yesterday's model A/B (benchmarks/bench_model_ab.py) used only 60 samples and
reported base rho=0.27 vs large rho=0.36 -- but with n=60 a ~0.09 rho gap is
not robustly distinguishable from noise. This script scales to 300 stratified
samples and adds 95% bootstrap confidence intervals so we can say whether the
large model's apparent edge is real.

Design
------
- 300 samples: 60 from each of 5 human_consistency RANK tiers (same rank-band
  stratification as Task 1/2, deterministic). SummEval consistency is heavily
  skewed to 5.0, so rank bands (not value quantiles) guarantee tail coverage.
- Score groundedness ONLY (the dimension human `consistency` measures) with:
    A) cross-encoder/nli-deberta-v3-base   (current default)
    B) cross-encoder/nli-deberta-v3-large
  both with top_k_premises=8.
- 95% bootstrap CI (1000 resamples, fixed seed) on Spearman rho for each model,
  resampling sample INDICES so each bootstrap replicate uses the same indices
  for both models (paired) -- and also a CI on the paired rho DIFFERENCE.
- Report whether the two per-model CIs overlap (no robust improvement) or are
  disjoint (robust), plus the difference-CI (the more powerful paired test).

Outputs:
  benchmarks/results/model_ab_powered.json
  benchmarks/results/model_ab_powered.md

Run:
  set PYTHONIOENCODING=utf-8
  python benchmarks/bench_model_ab_powered.py
  python benchmarks/bench_model_ab_powered.py --n 10   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np  # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402

from scroot.metrics.groundedness import score_groundedness  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
DATASET = Path(__file__).parent / "datasets" / "summeval.jsonl"
OUT_JSON = RESULTS_DIR / "model_ab_powered.json"
OUT_MD = RESULTS_DIR / "model_ab_powered.md"

EMB = "all-MiniLM-L6-v2"
TOP_K_PREMISES = 8
MODEL_A = "cross-encoder/nli-deberta-v3-base"
MODEL_B = "cross-encoder/nli-deberta-v3-large"

PER_TIER = 60
N_TIERS = 5
N_BOOT = 1000
SEED = 1234


def _chunk_article(text: str) -> list[str]:
    import re
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = [s.strip() for s in sents if len(s.split()) >= 5]
    return chunks if chunks else [text]


def _load() -> list[dict]:
    out = []
    with DATASET.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["summary_idx"] = int(d["summary_idx"])
            d["human_consistency"] = float(d["human_consistency"])
            out.append(d)
    return out


def _stratified_subset(records, per_tier):
    ordered = sorted(records, key=lambda r: (r["human_consistency"],
                                             r["doc_id"], r["summary_idx"]))
    n = len(ordered)
    band = n / N_TIERS
    sub = []
    for t in range(N_TIERS):
        lo, hi = int(round(t * band)), int(round((t + 1) * band))
        b = ordered[lo:hi]
        if not b:
            continue
        if len(b) <= per_tier:
            sub.extend(b)
        else:
            step = len(b) / per_tier
            sub.extend(b[int(i * step)] for i in range(per_tier))
    return sub


def _score_model(records, nli_model):
    scores, lats = [], []
    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        ctx = _chunk_article(rec["source"])
        ts = time.perf_counter()
        g, _ = score_groundedness(
            rec["summary"], ctx,
            nli_model=nli_model,
            embedding_model=EMB,
            top_k_chunks=3,
            top_k_premises=TOP_K_PREMISES,
        )
        lats.append((time.perf_counter() - ts) * 1000.0)
        scores.append(g)
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            eta = (len(records) - (i + 1)) / rate
            print(f"    {nli_model.split('/')[-1]} {i+1}/{len(records)}  "
                  f"{rate:.2f}/s  eta {eta/60:.1f}m", flush=True)
    return scores, lats


def _bootstrap_rho_ci(scores_a, scores_b, humans, n_boot, seed):
    """Paired bootstrap over sample indices.

    Returns per-model rho CIs and the CI on (rho_B - rho_A), all 95%.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    h = np.asarray(humans, dtype=float)
    n = len(h)
    boot_a, boot_b, boot_d = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        hh = h[idx]
        # Spearman is undefined if resample has zero variance; guard it.
        if np.ptp(hh) == 0:
            continue
        ra = spearmanr(a[idx], hh).correlation
        rb = spearmanr(b[idx], hh).correlation
        if np.isnan(ra) or np.isnan(rb):
            continue
        boot_a.append(ra)
        boot_b.append(rb)
        boot_d.append(rb - ra)
    boot_a = np.array(boot_a)
    boot_b = np.array(boot_b)
    boot_d = np.array(boot_d)

    def ci(arr):
        return (float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)))
    return ci(boot_a), ci(boot_b), ci(boot_d), len(boot_d)


def _block(scores, lats, humans):
    rho, prho = spearmanr(scores, humans)
    r, pr = pearsonr(scores, humans)
    return {
        "n": len(scores),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(f"{prho:.3g}"),
        "pearson_r": round(float(r), 4),
        "pearson_p": float(f"{pr:.3g}"),
        "mean_latency_ms": round(sum(lats) / len(lats), 1),
        "mean_groundedness": round(sum(scores) / len(scores), 4),
    }


def run(n_override):
    import datetime
    records = _load()
    subset = _stratified_subset(records, PER_TIER)
    if n_override is not None:
        subset = subset[:n_override]
    humans = [r["human_consistency"] for r in subset]
    print(f"Powered model A/B on {len(subset)} stratified SummEval samples "
          f"(groundedness only, top_k_premises={TOP_K_PREMISES})\n", flush=True)

    print(f"  A: {MODEL_A}", flush=True)
    a_scores, a_lats = _score_model(subset, MODEL_A)
    print(f"  B: {MODEL_B}", flush=True)
    b_scores, b_lats = _score_model(subset, MODEL_B)

    a = _block(a_scores, a_lats, humans)
    b = _block(b_scores, b_lats, humans)

    ci_a, ci_b, ci_diff, n_eff = _bootstrap_rho_ci(
        a_scores, b_scores, humans, N_BOOT, SEED)

    overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
    diff_excludes_zero = (ci_diff[0] > 0) or (ci_diff[1] < 0)

    deltas = [abs(x - y) for x, y in zip(a_scores, b_scores)]
    ab_rho = spearmanr(a_scores, b_scores).correlation

    payload = {
        "benchmark": "model_ab_powered_summeval_groundedness",
        "date": datetime.date.today().isoformat(),
        "n_samples": len(subset),
        "stratification": f"{N_TIERS} rank tiers x {PER_TIER}/tier",
        "top_k_premises": TOP_K_PREMISES,
        "bootstrap": {"n_iter": N_BOOT, "n_effective": n_eff, "seed": SEED,
                      "ci": "95% percentile, paired over sample indices"},
        "model_a": {"name": MODEL_A, **a,
                    "spearman_rho_ci95": [round(ci_a[0], 4), round(ci_a[1], 4)]},
        "model_b": {"name": MODEL_B, **b,
                    "spearman_rho_ci95": [round(ci_b[0], 4), round(ci_b[1], 4)]},
        "rho_difference_b_minus_a": {
            "point": round(b["spearman_rho"] - a["spearman_rho"], 4),
            "ci95": [round(ci_diff[0], 4), round(ci_diff[1], 4)],
            "excludes_zero": bool(diff_excludes_zero),
        },
        "per_model_ci_overlap": bool(overlap),
        "agreement": {
            "mean_abs_groundedness_delta": round(sum(deltas) / len(deltas), 4),
            "inter_model_spearman": round(float(ab_rho), 4),
        },
        "robust_improvement": bool(diff_excludes_zero),
        "conclusion": (
            "Large model is a ROBUST improvement (paired rho-difference CI "
            "excludes 0)." if diff_excludes_zero else
            "No robust improvement: the paired rho-difference 95% CI includes "
            "0, so the large model's apparent edge is within sampling noise."),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(payload)

    print("\n=== POWERED MODEL A/B ===")
    print(f"  A base  rho={a['spearman_rho']:+.4f} "
          f"CI[{ci_a[0]:.3f},{ci_a[1]:.3f}]  lat={a['mean_latency_ms']:.0f}ms")
    print(f"  B large rho={b['spearman_rho']:+.4f} "
          f"CI[{ci_b[0]:.3f},{ci_b[1]:.3f}]  lat={b['mean_latency_ms']:.0f}ms")
    print(f"  diff (B-A)={payload['rho_difference_b_minus_a']['point']:+.4f} "
          f"CI[{ci_diff[0]:.3f},{ci_diff[1]:.3f}] "
          f"excludes_zero={diff_excludes_zero}")
    print(f"  per-model CIs overlap: {overlap}")
    print(f"  -> {payload['conclusion']}")
    print(f"Saved -> {OUT_JSON}")


def _write_md(p):
    a, b = p["model_a"], p["model_b"]
    d = p["rho_difference_b_minus_a"]
    lines = []
    lines.append("# Powered Model A/B: NLI base vs large (Task 4)")
    lines.append("")
    lines.append(
        f"SummEval groundedness vs human `consistency`, "
        f"**{p['n_samples']} stratified samples** "
        f"({p['stratification']}), top_k_premises={p['top_k_premises']}. "
        f"95% bootstrap CIs from {p['bootstrap']['n_iter']} paired resamples "
        f"(seed {p['bootstrap']['seed']})."
    )
    lines.append("")
    lines.append("| Model | Spearman rho | 95% CI | Pearson r | mean latency |")
    lines.append("|-------|-------------|--------|-----------|--------------|")
    lines.append(
        f"| A: nli-deberta-v3-base (default) | {a['spearman_rho']:.4f} | "
        f"[{a['spearman_rho_ci95'][0]:.3f}, {a['spearman_rho_ci95'][1]:.3f}] | "
        f"{a['pearson_r']:.4f} | {a['mean_latency_ms']:.0f} ms |")
    lines.append(
        f"| B: nli-deberta-v3-large | {b['spearman_rho']:.4f} | "
        f"[{b['spearman_rho_ci95'][0]:.3f}, {b['spearman_rho_ci95'][1]:.3f}] | "
        f"{b['pearson_r']:.4f} | {b['mean_latency_ms']:.0f} ms |")
    lines.append("")
    lines.append(
        f"**Paired rho difference (B - A):** {d['point']:+.4f}, "
        f"95% CI [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] -- "
        f"{'EXCLUDES' if d['excludes_zero'] else 'INCLUDES'} zero.")
    lines.append("")
    lines.append(
        f"**Per-model CI overlap:** "
        f"{'YES' if p['per_model_ci_overlap'] else 'NO'}.")
    lines.append("")
    lines.append(f"**Conclusion:** {p['conclusion']}")
    lines.append("")
    lines.append(
        f"Inter-model agreement: mean |delta groundedness| = "
        f"{p['agreement']['mean_abs_groundedness_delta']}, "
        f"inter-model Spearman = {p['agreement']['inter_model_spearman']}.")
    lines.append("")
    lines.append(
        "Note: the paired rho-difference CI is the powered test of record; "
        "per-model CIs ignore the fact that both models score the same samples "
        "and so overlap even when the paired difference is significant.")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="cap sample count (smoke)")
    args = ap.parse_args()
    run(args.n)


if __name__ == "__main__":
    main()
