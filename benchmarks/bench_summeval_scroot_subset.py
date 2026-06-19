"""
Score scroot on a stratified SummEval subset, saving FULL per-dimension scores.

The original full-1600 run (bench_summeval.py) reported aggregate correlations
(groundedness rho=0.36, IQS rho=0.12) but its per-sample detail in
summeval_results.json was later overwritten by a 3-sample smoke run. Task 2's
applicability-gating fix needs real per-dimension scores to demonstrate the
IQS-vs-human_consistency improvement on data, not just the n=3 smoke rows.

This re-scores a deterministic 400-sample stratified subset (80 per
human_consistency rank-band x 5 bands) and writes every dimension score to
summeval_subset_scores.json, so bench_composite_fix_validate.py can recompute
gated vs ungated IQS correlation honestly.

Run:
  set PYTHONIOENCODING=utf-8
  python benchmarks/bench_summeval_scroot_subset.py
  python benchmarks/bench_summeval_scroot_subset.py --max 50   # smoke
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

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "datasets" / "summeval.jsonl"
OUT = RESULTS_DIR / "summeval_subset_scores.json"

QUERY = "Summarize the following article."
SUBSET_PER_TIER = 80
N_TIERS = 5


def _chunk_article(text: str) -> list[str]:
    import re
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = [s.strip() for s in sents if len(s.split()) >= 5]
    return chunks if chunks else [text]


def _load() -> list[dict]:
    records = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["summary_idx"] = int(d["summary_idx"])
            for k in ("human_consistency", "human_relevance",
                      "human_coherence", "human_fluency"):
                d[k] = float(d[k])
            records.append(d)
    return records


def _stratified_subset(records: list[dict], per_tier: int) -> list[dict]:
    ordered = sorted(
        records,
        key=lambda r: (r["human_consistency"], r["doc_id"], r["summary_idx"]),
    )
    n = len(ordered)
    band_size = n / N_TIERS
    subset: list[dict] = []
    for tier in range(N_TIERS):
        lo = int(round(tier * band_size))
        hi = int(round((tier + 1) * band_size))
        band = ordered[lo:hi]
        if not band:
            continue
        if len(band) <= per_tier:
            subset.extend(band)
        else:
            step = len(band) / per_tier
            subset.extend(band[int(i * step)] for i in range(per_tier))
    return subset


def run(max_samples: int | None) -> None:
    import datetime
    from scroot import Auditor

    records = _load()
    subset = _stratified_subset(records, SUBSET_PER_TIER)
    if max_samples is not None:
        subset = subset[:max_samples]
    print(f"Scoring {len(subset)} stratified SummEval samples with scroot...",
          flush=True)

    auditor = Auditor()
    out = []
    t0 = time.perf_counter()
    for i, rec in enumerate(subset):
        ts = time.perf_counter()
        r = auditor.score(query=QUERY, response=rec["summary"],
                          context=_chunk_article(rec["source"]))
        lat = (time.perf_counter() - ts) * 1000.0
        out.append({
            "doc_id": rec["doc_id"],
            "summary_idx": rec["summary_idx"],
            "scroot_iqs": r.iqs,
            "scroot_groundedness": r.groundedness if r.groundedness is not None else 0.0,
            "scroot_completeness": r.completeness,
            "scroot_relevance": r.relevance,
            "scroot_consistency": r.consistency,
            "scroot_confidence": r.confidence,
            "scroot_latency_ms": round(lat, 1),
            "human_consistency": rec["human_consistency"],
            "human_relevance": rec["human_relevance"],
            "human_coherence": rec["human_coherence"],
            "human_fluency": rec["human_fluency"],
        })
        if (i + 1) % 25 == 0:
            el = time.perf_counter() - t0
            rate = (i + 1) / el
            eta = (len(subset) - i - 1) / rate
            print(f"  {i+1}/{len(subset)}  {rate:.2f}/s  ETA {eta/60:.1f} min",
                  flush=True)

    lat = [r["scroot_latency_ms"] for r in out]
    payload = {
        "benchmark": "summeval_subset_scroot",
        "date": datetime.date.today().isoformat(),
        "n_samples": len(out),
        "query": QUERY,
        "scroot_mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
        "per_sample_scores": out,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min -> {OUT}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None)
    args = p.parse_args()
    run(args.max)


if __name__ == "__main__":
    main()
