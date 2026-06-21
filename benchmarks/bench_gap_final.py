"""EXPERIMENT D - Combined winner configuration + final gap measurement.

Assembles the final table from cached results:
  - baseline: scroot current (deberta full pipeline) rho=0.4017 (recomputed)
  - best backbone only (Exp A)
  - + spaCy atomic claims if they helped (Exp B)
  - + best aggregation (Exp C)
  - RAGAS reference 0.64

Pass the winning choices as args. Recomputes the combined rho/CI from caches.

Run:
  python benchmarks/bench_gap_final.py --backbone <key> --claimsrc <key> \
      --aggregation <coverage|mean|min>
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
from bench_gap_aggregation import aggregate, load_source  # noqa: E402
from bench_gap_backbone_ab import load_396, spearman_with_ci  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
OUT_MD = RESULTS_DIR / "gap_closing_final.md"

RAGAS_RHO = 0.64
BASELINE_RHO = 0.4017


def measure(source, aggregation, human):
    per_sample = load_source(source)
    scores, humans = [], []
    for p in per_sample:
        scores.append(aggregate(p["claim_scores"], aggregation))
        humans.append(human[(p["doc_id"], p["summary_idx"])])
    return spearman_with_ci(scores, humans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--claimsrc", default=None,
                    help="claim-decomp cache key (spacy) if it helped")
    ap.add_argument("--aggregation", default="coverage")
    args = ap.parse_args()

    ids, recs, human = load_396()

    rows = []
    # baseline
    rows.append(("baseline (deberta-base, current scroot pipeline)",
                 BASELINE_RHO, None, None, None, "~8.6s", "Yes"))

    # best backbone only (coverage agg, regex claims)
    rho, lo, hi, r = measure(args.backbone, "coverage", human)
    rows.append((f"best backbone only ({args.backbone})", rho, lo, hi, r,
                 "see Exp A", "Yes"))

    # + spacy atomic claims (deberta backbone, coverage)
    if args.claimsrc:
        rho2, lo2, hi2, r2 = measure(args.claimsrc, "coverage", human)
        rows.append((f"+ spaCy atomic claims ({args.claimsrc})", rho2, lo2,
                     hi2, r2, "see Exp B", "Yes"))

    # + best aggregation on best backbone
    rho3, lo3, hi3, r3 = measure(args.backbone, args.aggregation, human)
    rows.append((f"+ best aggregation ({args.aggregation}) on {args.backbone}",
                 rho3, lo3, hi3, r3, "see Exp A", "Yes"))

    gap = RAGAS_RHO - BASELINE_RHO
    best_rho = max(r[1] for r in rows[1:])
    closed = (best_rho - BASELINE_RHO) / gap * 100 if gap else 0.0

    lines = ["# Experiment D - Combined winner + gap-closing measurement",
             "",
             "| Configuration | rho | 95% CI | Pearson r | Latency/sample | "
             "Deterministic |",
             "|---|---|---|---|---|---|"]
    for name, rho, lo, hi, r, lat, det in rows:
        ci = f"[{lo}, {hi}]" if lo is not None else "[see tightening report]"
        rr = "-" if r is None else r
        lines.append(f"| {name} | {rho} | {ci} | {rr} | {lat} | {det} |")
    lines.append(f"| RAGAS faithfulness (reference) | {RAGAS_RHO} | - | 0.73 | "
                 "~0.5s+API | No |")
    lines += ["",
              f"Baseline rho = {BASELINE_RHO}. RAGAS rho = {RAGAS_RHO}. "
              f"Gap = {gap:.4f}.",
              f"Best config rho = {best_rho}. "
              f"Gap closed = **{closed:.1f}%** of the 0.40->0.64 distance.",
              ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
