"""EXPERIMENT A - Grounding backbone A/B with purpose-built factuality models.

Scores groundedness on the same 396 SummEval samples using several backbones
that all plug into ONE retrieval+aggregation harness:

  - cross-encoder/nli-deberta-v3-base   (NLI 3-class, scroot baseline backbone)
  - lytang/MiniCheck-RoBERTa-Large      (binary support classifier)
  - lytang/MiniCheck-Flan-T5-Large      (seq2seq support classifier)
  - yzha/AlignScore (if its package + checkpoint are available)

Harness (mirrors src/scroot groundedness retrieval, backbone-agnostic):
  1. extract atomic claims from the summary
  2. embed claims + context sentences once (all-MiniLM-L6-v2)
  3. per claim: retrieve top_k_premises=8 most similar context sentences
  4. run the backbone on those (premise, claim) pairs -> support prob
  5. claim grounded if max support prob >= 0.5
  6. response score = fraction of grounded claims (coverage ratio, scroot default)

Outputs per-claim scores to a JSON cache (reused by Exp B/C/D) and writes
benchmarks/results/grounding_backbone_ab.md.

Run:
  $env:PYTHONIOENCODING="utf-8"; python benchmarks/bench_gap_backbone_ab.py
  ... --max 20   (smoke)
  ... --models deberta,minicheck_roberta   (subset)
"""

from __future__ import annotations

import argparse
import json
import re
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

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "datasets" / "summeval.jsonl"
COMPETITORS = RESULTS_DIR / "summeval_competitors.json"
CACHE = RESULTS_DIR / "gap_backbone_claim_scores.json"
OUT_MD = RESULTS_DIR / "grounding_backbone_ab.md"

TOP_K_PREMISES = 8
ENTAIL_THRESHOLD = 0.5
EMB_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_396():
    comp = json.load(open(COMPETITORS, encoding="utf-8"))
    ids = [(p["doc_id"], p["summary_idx"])
           for p in comp["deepeval_raw"]["per_sample"]]
    recs = {}
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            recs[(d["doc_id"], int(d["summary_idx"]))] = d
    human = {(p["doc_id"], p["summary_idx"]): p["human_consistency"]
             for p in comp["deepeval_raw"]["per_sample"]}
    return ids, recs, human


def chunk_article(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = [s.strip() for s in sents if len(s.split()) >= 5]
    return chunks if chunks else [text]


def context_sentences(context: list[str]) -> list[str]:
    """Sentence-split each context chunk (premise granularity)."""
    out = []
    for chunk in context:
        sents = re.split(r"(?<=[.!?])\s+", chunk.strip())
        sents = [s.strip() for s in sents if len(s.split()) >= 4]
        out.extend(sents if sents else [chunk])
    return out


# --------------------------------------------------------------------------
# Backbones - each exposes score_pairs(pairs) -> list[float] support-prob
# --------------------------------------------------------------------------
class DebertaBackbone:
    """3-class NLI; support prob = entailment probability."""
    name = "deberta-base"
    hf = "cross-encoder/nli-deberta-v3-base"
    size = "184M"

    def __init__(self, device="cpu"):
        from scroot.metrics._utils import softmax
        from scroot.models import get_nli_model
        self._softmax = softmax
        self.model = get_nli_model(self.hf, device=device)

    def score_pairs(self, pairs):
        # pairs are (premise, claim)
        raw = self.model.predict(pairs)
        # deberta-v3 label order: 0=contradiction,1=entailment,2=neutral
        out = []
        for row in raw:
            p = self._softmax(row)
            out.append(float(p[1]))
        return out


class MiniCheckRoberta:
    name = "minicheck-roberta-large"
    hf = "lytang/MiniCheck-RoBERTa-Large"
    size = "355M"

    def __init__(self, device="cpu"):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.hf)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf).to(device).eval()

    def score_pairs(self, pairs):
        out = []
        bs = 16
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            docs = [p[0] for p in batch]
            claims = [p[1] for p in batch]
            enc = self.tok(docs, claims, truncation=True, max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
                probs = self.torch.softmax(logits, dim=-1)[:, 1]
            out.extend(probs.cpu().tolist())
        return out


class MiniCheckFlanT5:
    name = "minicheck-flan-t5-large"
    hf = "lytang/MiniCheck-Flan-T5-Large"
    size = "770M"

    def __init__(self, device="cpu"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.hf)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hf).to(device).eval()

    def score_pairs(self, pairs):
        # MiniCheck Flan-T5 prompt format: "predict: document: <doc> claim: <claim>"
        out = []
        bs = 8
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            texts = [self._fmt(p[0], p[1]) for p in batch]
            enc = self.tok(texts, truncation=True, max_length=2048,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                dec = self.torch.zeros(
                    (enc["input_ids"].size(0), 1),
                    dtype=self.torch.long).to(self.device)
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=dec).logits.squeeze(1)
                # tokens 3='0'(no support), 209='1'(support)
                label_logits = logits[:, self.torch.tensor([3, 209])]
                probs = self.torch.softmax(label_logits, dim=-1)[:, 1]
            out.extend(probs.cpu().tolist())
        return out

    @staticmethod
    def _fmt(doc, claim):
        return f"predict: {doc}\nclaim: {claim}"


BACKBONES = {
    "deberta": DebertaBackbone,
    "minicheck_roberta": MiniCheckRoberta,
    "minicheck_flan_t5": MiniCheckFlanT5,
}


# --------------------------------------------------------------------------
# Scoring harness
# --------------------------------------------------------------------------
def score_dataset(backbone, ids, recs, emb_model, max_samples=None):
    from scroot.text_utils import extract_atomic_claims
    ids = ids if max_samples is None else ids[:max_samples]
    per_sample = []
    lats = []
    for n, key in enumerate(ids):
        rec = recs[key]
        t0 = time.perf_counter()
        claims = extract_atomic_claims(rec["summary"])
        ctx = context_sentences(chunk_article(rec["source"]))
        if not claims:
            per_sample.append({"doc_id": key[0], "summary_idx": key[1],
                               "claim_scores": [], "score": 1.0})
            lats.append((time.perf_counter() - t0) * 1000)
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
            "score": grounded / len(claims),
        })
        lats.append((time.perf_counter() - t0) * 1000)
        if (n + 1) % 25 == 0:
            print(f"    {backbone.name}: {n+1}/{len(ids)}  "
                  f"{np.mean(lats):.0f}ms/sample", flush=True)
    return per_sample, float(np.mean(lats))


def spearman_with_ci(x, y, n_boot=1000, seed=0):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho = spearmanr(x, y).correlation
    r = pearsonr(x, y)[0]
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rb = spearmanr(x[idx], y[idx]).correlation
        if not np.isnan(rb):
            boots.append(rb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(rho, 4), round(float(lo), 4), round(float(hi), 4), round(r, 4)


def determinism_check(backbone, recs, ids, emb_model, n_samples=10, repeats=10):
    sub = ids[:n_samples]
    runs = []
    for _ in range(repeats):
        ps, _ = score_dataset(backbone, sub, recs, emb_model)
        runs.append([round(p["score"], 6) for p in ps])
    base = runs[0]
    deviations = sum(1 for r in runs for a, b in zip(base, r) if a != b)
    return deviations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--models", type=str,
                    default="deberta,minicheck_roberta,minicheck_flan_t5")
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()

    ids, recs, human = load_396()
    print(f"Loaded {len(ids)} samples.", flush=True)

    from scroot.models import get_embedding_model
    emb_model = get_embedding_model(EMB_MODEL, device="cpu")

    cache = {}
    if CACHE.exists():
        cache = json.load(open(CACHE, encoding="utf-8"))

    results = {}
    for mkey in args.models.split(","):
        mkey = mkey.strip()
        if mkey not in BACKBONES:
            print(f"  skip unknown model {mkey}")
            continue
        print(f"\n=== {mkey} ===", flush=True)
        try:
            t0 = time.perf_counter()
            backbone = BACKBONES[mkey]()
            load_t = time.perf_counter() - t0
            print(f"  loaded in {load_t:.1f}s", flush=True)
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            results[mkey] = {"error": str(e)[:200]}
            continue

        per_sample, mean_lat = score_dataset(backbone, ids, recs, emb_model,
                                             args.max)
        cur_ids = ids if args.max is None else ids[:args.max]
        scores = [p["score"] for p in per_sample]
        humans = [human[(p["doc_id"], p["summary_idx"])] for p in per_sample]
        rho, lo, hi, r = spearman_with_ci(scores, humans)

        det = None
        if not args.skip_determinism:
            det = determinism_check(backbone, recs, cur_ids, emb_model)

        results[mkey] = {
            "name": backbone.name, "hf": backbone.hf, "size": backbone.size,
            "rho": rho, "ci_lo": lo, "ci_hi": hi, "pearson": r,
            "mean_latency_ms": round(mean_lat, 1),
            "determinism_deviations": det,
        }
        cache[backbone.name] = {
            "hf": backbone.hf,
            "per_sample": per_sample,
        }
        print(f"  rho={rho} CI[{lo},{hi}] pearson={r} "
              f"lat={mean_lat:.0f}ms det_dev={det}", flush=True)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), indent=2)

    # baseline reference
    baseline = {"name": "scroot-current (deberta, full pipeline)",
                "rho": 0.4017, "ci": "see deberta row", "size": "184M"}

    write_md(results, baseline, args)
    print(f"\nWrote {OUT_MD}")


def write_md(results, baseline, args):
    lines = ["# Experiment A - Grounding backbone A/B",
             "",
             f"Same 396 SummEval samples (or --max {args.max}). "
             f"Harness: atomic claims, top-{TOP_K_PREMISES} premise retrieval, "
             f"max support prob per claim, coverage-ratio aggregation "
             f"(threshold {ENTAIL_THRESHOLD}).",
             "",
             "Backbone support-prob mapping: deberta = P(entailment); "
             "MiniCheck = P(supported).",
             "",
             "| Model | rho | 95% CI | Pearson r | Latency/sample | Size | "
             "Air-gap | Det. dev |",
             "|---|---|---|---|---|---|---|---|"]
    # scroot current baseline (production pipeline) as reference row
    lines.append("| scroot current (deberta, full pipeline) | 0.4017 | "
                 "[recomputed below] | 0.3901 | ~8.6s* | 184M | yes | 0 |")
    for mkey, r in results.items():
        if "error" in r:
            lines.append(f"| {mkey} | FAILED | - | - | - | - | - | - | "
                         f"({r['error']}) |")
            continue
        lines.append(
            f"| {r['name']} | {r['rho']} | [{r['ci_lo']}, {r['ci_hi']}] | "
            f"{r['pearson']} | {r['mean_latency_ms']/1000:.2f}s | {r['size']} | "
            f"yes | {r['determinism_deviations']} |")
    lines += ["",
              "*scroot current full-pipeline latency from prior 1600 run "
              "(includes all dimensions + fallback). Harness latency is "
              "groundedness-only and not directly comparable.",
              "",
              "Notes:",
              "- All backbones are classifiers/cross-encoders (deterministic, "
              "no generative sampling). Det. dev = deviations across 10x10 "
              "repeat check (0 required).",
              "- Air-gap: all run fully local on CPU after one-time HF "
              "download; $0 API cost.",
              ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
