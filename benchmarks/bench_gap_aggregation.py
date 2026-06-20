"""EXPERIMENT C - Aggregation formula comparison.

Recomputes groundedness from per-claim support probabilities already produced
by Experiments A and B (no model re-scoring). Three aggregations:

  1. coverage   : fraction of claims with support prob >= 0.5  (scroot current)
  2. mean       : mean support prob across all claims
  3. min        : minimum support prob (weakest-link / strict faithfulness)

Compares each against human_consistency on the 396. Source of per-claim
scores selectable via --source (best backbone cache key or claim-method key).

Run:
  $env:PYTHONIOENCODING="utf-8"; python benchmarks/bench_gap_aggregation.py --source <key>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
from bench_gap_backbone_ab import load_396, spearman_with_ci  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
BACKBONE_CACHE = RESULTS_DIR / "gap_backbone_claim_scores.json"
DECOMP_CACHE = RESULTS_DIR / "gap_claim_decomp_scores.json"
OUT_MD = RESULTS_DIR / "aggregation_comparison.md"

THRESH = 0.5


def aggregate(claim_scores, method):
    if not claim_scores:
        return 1.0  # no claims -> vacuously grounded (matches harness)
    a = np.asarray(claim_scores, float)
    if method == "coverage":
        return float(np.mean(a >= THRESH))
    if method == "mean":
        return float(np.mean(a))
    if method == "min":
        return float(np.min(a))
    raise ValueError(method)


def load_source(source):
    for cache in (BACKBONE_CACHE, DECOMP_CACHE):
        if cache.exists():
            data = json.load(open(cache, encoding="utf-8"))
            if source in data:
                return data[source]["per_sample"]
    raise SystemExit(f"source {source!r} not found in caches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="cache key, e.g. deberta-base / minicheck-roberta-large / spacy")
    args = ap.parse_args()

    ids, recs, human = load_396()
    per_sample = load_source(args.source)
    hlut = human

    rows = {}
    for method in ("coverage", "mean", "min"):
        scores, humans = [], []
        for p in per_sample:
            scores.append(aggregate(p["claim_scores"], method))
            humans.append(hlut[(p["doc_id"], p["summary_idx"])])
        rho, lo, hi, r = spearman_with_ci(scores, humans)
        rows[method] = (rho, lo, hi, r)
        print(f"  {method:9s} rho={rho} CI[{lo},{hi}] r={r}", flush=True)

    best = max(rows, key=lambda m: rows[m][0])
    label = {"coverage": "coverage ratio >=0.5 (scroot current)",
             "mean": "mean support prob", "min": "min support prob"}
    lines = [f"# Experiment C - Aggregation formula comparison",
             "",
             f"Source per-claim scores: `{args.source}`. Same 396 samples. "
             f"No re-scoring - aggregations recomputed from cached per-claim "
             f"support probabilities.",
             "",
             "| Aggregation | rho | 95% CI | Pearson r |",
             "|---|---|---|---|"]
    for m in ("coverage", "mean", "min"):
        rho, lo, hi, r = rows[m]
        star = "  **<- best**" if m == best else ""
        lines.append(f"| {label[m]} | {rho}{star} | [{lo}, {hi}] | {r} |")
    lines += ["", f"Best aggregation by rho: **{label[best]}** "
              f"(rho={rows[best][0]})."]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nBest: {best} (rho={rows[best][0]}). Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
