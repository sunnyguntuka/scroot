"""
Task 4 - NLI model A/B on SummEval groundedness.

Compares two NLI cross-encoders as the groundedness backbone, scoring ONLY the
groundedness dimension (the one SummEval's human ``consistency`` annotation
measures):

  A) cross-encoder/nli-deberta-v3-base   (default, ~180M params)
  B) cross-encoder/nli-deberta-v3-large  (~435M params)

Metrics per model: Spearman / Pearson vs human_consistency, mean latency, and
inter-model agreement (mean |B - A| groundedness delta, rank correlation).

Subset
------
Uses the SAME deterministic 400-sample stratified subset as Task 2/Task 3 so the
comparison is apples-to-apples. Scoring all 1600 with the large model is
~10h (base alone is 229 min); 400 stratified samples is statistically ample for
a backbone A/B and keeps the sprint tractable. Pass --n to override; --full
scores all 1600.

Uses the Task 3 optimisation (top_k_premises) so the large model stays
affordable. groundedness-only: completeness/relevance/consistency/confidence are
not computed here.

Run:
  set PYTHONIOENCODING=utf-8
  python benchmarks/bench_model_ab.py
  python benchmarks/bench_model_ab.py --n 50   # smoke
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

from scroot.metrics.groundedness import score_groundedness  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
DATASET = Path(__file__).parent / "datasets" / "summeval.jsonl"
OUT = RESULTS_DIR / "model_ab.json"

QUERY = "Summarize the following article."
EMB = "all-MiniLM-L6-v2"
TOP_K_PREMISES = 8

MODEL_A = "cross-encoder/nli-deberta-v3-base"
MODEL_B = "cross-encoder/nli-deberta-v3-large"

SUBSET_PER_TIER = 80
N_TIERS = 5


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


def _stratified_subset(records: list[dict], per_tier: int) -> list[dict]:
    ordered = sorted(records, key=lambda r: (r["human_consistency"],
                                             r["doc_id"], r["summary_idx"]))
    n = len(ordered)
    band = n / N_TIERS
    sub: list[dict] = []
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


def _spearman(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)


def _pearson(x, y):
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return float(r), float(p)


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
            print(f"    {nli_model.split('/')[-1]} {i+1}/{len(records)}  "
                  f"{rate:.2f}/s", flush=True)
    return scores, lats


def _block(scores, lats, humans):
    rho, prho = _spearman(scores, humans)
    r, pr = _pearson(scores, humans)
    return {
        "n": len(scores),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(prho, 6),
        "pearson_r": round(r, 4),
        "pearson_p": round(pr, 6),
        "mean_latency_ms": round(sum(lats) / len(lats), 1),
        "mean_groundedness": round(sum(scores) / len(scores), 4),
    }


def run(n: int | None, full: bool) -> None:
    import datetime
    records = _load()
    if full:
        subset = records
    else:
        subset = _stratified_subset(records, SUBSET_PER_TIER)
    if n is not None:
        subset = subset[:n]
    humans = [r["human_consistency"] for r in subset]
    print(f"Model A/B on {len(subset)} SummEval samples "
          f"(groundedness only, top_k_premises={TOP_K_PREMISES})\n", flush=True)

    print(f"  A: {MODEL_A}", flush=True)
    a_scores, a_lats = _score_model(subset, MODEL_A)
    print(f"  B: {MODEL_B}", flush=True)
    b_scores, b_lats = _score_model(subset, MODEL_B)

    a = _block(a_scores, a_lats, humans)
    b = _block(b_scores, b_lats, humans)
    deltas = [abs(x - y) for x, y in zip(a_scores, b_scores)]
    ab_rho, _ = _spearman(a_scores, b_scores)

    payload = {
        "benchmark": "model_ab_summeval_groundedness",
        "date": datetime.date.today().isoformat(),
        "n_samples": len(subset),
        "subset": ("full-1600" if full
                   else f"stratified-{len(subset)}"),
        "top_k_premises": TOP_K_PREMISES,
        "model_a": {"name": MODEL_A, **a},
        "model_b": {"name": MODEL_B, **b},
        "agreement": {
            "mean_abs_groundedness_delta": round(sum(deltas) / len(deltas), 4),
            "max_abs_delta": round(max(deltas), 4),
            "inter_model_spearman": round(ab_rho, 4),
        },
        "winner_by_rho": ("B" if b["spearman_rho"] > a["spearman_rho"]
                          else "A" if a["spearman_rho"] > b["spearman_rho"]
                          else "tie"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Model A/B (groundedness vs human_consistency) ===")
    print(f"  A base  rho={a['spearman_rho']:+.4f}  r={a['pearson_r']:+.4f}  "
          f"lat={a['mean_latency_ms']:.0f}ms")
    print(f"  B large rho={b['spearman_rho']:+.4f}  r={b['pearson_r']:+.4f}  "
          f"lat={b['mean_latency_ms']:.0f}ms")
    print(f"  agreement: mean|delta|={payload['agreement']['mean_abs_groundedness_delta']}  "
          f"inter-model rho={payload['agreement']['inter_model_spearman']}")
    print(f"  winner by rho: {payload['winner_by_rho']}")
    print(f"Saved -> {OUT}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--full", action="store_true", help="Score all 1600 (slow).")
    args = p.parse_args()
    run(args.n, args.full)


if __name__ == "__main__":
    main()
