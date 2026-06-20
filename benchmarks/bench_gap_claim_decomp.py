"""EXPERIMENT B - Atomic claim decomposition via spaCy.

Re-scores the 396 SummEval samples with the deberta-base backbone, comparing
claim extraction methods:
  - regex      : current scroot extract_atomic_claims (compound-conj/semicolon)
  - spacy      : dependency-parse atomic decomposition (benchmarks/_spacy_claims)

Same retrieval + coverage-ratio aggregation harness as Experiment A. Writes
per-claim scores to gap_claim_decomp_scores.json (reused by Exp C/D) and
benchmarks/results/claim_decomposition.md.

Run:
  $env:PYTHONIOENCODING="utf-8"; python benchmarks/bench_gap_claim_decomp.py
  ... --max 20  (smoke)
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
sys.path.insert(0, str(Path(__file__).parent))  # for _spacy_claims

import numpy as np  # noqa: E402

from bench_gap_backbone_ab import (  # noqa: E402
    DebertaBackbone, ENTAIL_THRESHOLD, TOP_K_PREMISES, EMB_MODEL,
    chunk_article, context_sentences, load_396, spearman_with_ci)

RESULTS_DIR = Path(__file__).parent / "results"
CACHE = RESULTS_DIR / "gap_claim_decomp_scores.json"
OUT_MD = RESULTS_DIR / "claim_decomposition.md"


def score_with_claimfn(backbone, claim_fn, ids, recs, emb_model, human,
                       max_samples=None):
    ids = ids if max_samples is None else ids[:max_samples]
    per_sample = []
    lats = []
    claim_counts = []
    for n, key in enumerate(ids):
        rec = recs[key]
        t0 = time.perf_counter()
        claims = claim_fn(rec["summary"])
        ctx = context_sentences(chunk_article(rec["source"]))
        if not claims:
            per_sample.append({"doc_id": key[0], "summary_idx": key[1],
                               "claim_scores": [], "score": 1.0, "n_claims": 0})
            lats.append((time.perf_counter() - t0) * 1000)
            claim_counts.append(0)
            continue
        ctx_emb = emb_model.encode(ctx, convert_to_numpy=True)
        claim_emb = emb_model.encode(claims, convert_to_numpy=True)
        cn = np.linalg.norm(ctx_emb, axis=1) + 1e-8
        claim_scores = []
        for ci, claim in enumerate(claims):
            v = claim_emb[ci]
            sims = ctx_emb @ v / (cn * (np.linalg.norm(v) + 1e-8))
            k = min(TOP_K_PREMISES, len(ctx))
            top = np.argsort(sims)[::-1][:k]
            pairs = [(ctx[j], claim) for j in top]
            probs = backbone.score_pairs(pairs)
            claim_scores.append(float(max(probs)) if probs else 0.0)
        grounded = sum(1 for s in claim_scores if s >= ENTAIL_THRESHOLD)
        per_sample.append({
            "doc_id": key[0], "summary_idx": key[1],
            "claim_scores": [round(s, 4) for s in claim_scores],
            "score": grounded / len(claims), "n_claims": len(claims),
        })
        lats.append((time.perf_counter() - t0) * 1000)
        claim_counts.append(len(claims))
        if (n + 1) % 25 == 0:
            print(f"    {n+1}/{len(ids)}  {np.mean(lats):.0f}ms/sample  "
                  f"{np.mean(claim_counts):.1f} claims/resp", flush=True)
    return per_sample, float(np.mean(lats)), float(np.mean(claim_counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    # bench_gap_backbone_ab removes benchmarks/ from sys.path to avoid the
    # local datasets/ shadow. Re-add it briefly just for this local module,
    # then remove it again before any sentence-transformers imports.
    import sys
    _this_dir = str(Path(__file__).parent)
    sys.path.insert(0, _this_dir)
    from _spacy_claims import extract_atomic_claims_spacy
    sys.path.remove(_this_dir)

    ids, recs, human = load_396()
    from scroot.models import get_embedding_model
    from scroot.text_utils import extract_atomic_claims
    emb_model = get_embedding_model(EMB_MODEL, device="cpu")
    backbone = DebertaBackbone()

    methods = {
        "regex": extract_atomic_claims,
        "spacy": extract_atomic_claims_spacy,
    }

    cache = {}
    if CACHE.exists():
        cache = json.load(open(CACHE, encoding="utf-8"))

    results = {}
    for mname, fn in methods.items():
        print(f"\n=== claim method: {mname} ===", flush=True)
        per_sample, lat, cpr = score_with_claimfn(
            backbone, fn, ids, recs, emb_model, human, args.max)
        scores = [p["score"] for p in per_sample]
        humans = [human[(p["doc_id"], p["summary_idx"])] for p in per_sample]
        rho, lo, hi, r = spearman_with_ci(scores, humans)
        results[mname] = {"rho": rho, "ci_lo": lo, "ci_hi": hi, "pearson": r,
                          "latency_ms": round(lat, 1), "claims_per_resp": round(cpr, 2)}
        cache[mname] = {"per_sample": per_sample}
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), indent=2)
        print(f"  rho={rho} CI[{lo},{hi}] r={r} lat={lat:.0f}ms "
              f"claims/resp={cpr:.2f}", flush=True)

    lines = ["# Experiment B - Atomic claim decomposition",
             "",
             f"deberta-base backbone, top-{TOP_K_PREMISES} premise retrieval, "
             f"coverage-ratio aggregation. Same 396 samples"
             + (f" (--max {args.max})" if args.max else "") + ".",
             "",
             "| Claim method | rho | 95% CI | Pearson r | Latency/sample | "
             "Claims/response (mean) |",
             "|---|---|---|---|---|---|"]
    label = {"regex": "regex atomic (current scroot)",
             "spacy": "spaCy dependency atomic (new)"}
    for m, r in results.items():
        lines.append(f"| {label[m]} | {r['rho']} | [{r['ci_lo']}, {r['ci_hi']}] "
                     f"| {r['pearson']} | {r['latency_ms']/1000:.2f}s | "
                     f"{r['claims_per_resp']} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
