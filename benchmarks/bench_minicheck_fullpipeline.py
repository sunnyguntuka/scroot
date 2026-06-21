"""MiniCheck full-pipeline integration and latency validation.

Validates MiniCheck-RoBERTa-Large through the real Auditor.score() path:
  - End-to-end integrity across 100+ diverse inputs (Set A)
  - Only groundedness differs; other 4 dimensions must be identical
  - Full-pipeline determinism: 20 inputs x 10 passes, 0 deviations
  - Latency comparison on ~300 inputs (Set B), by input type

Run:
  $env:PYTHONIOENCODING="utf-8"
  python benchmarks/bench_minicheck_fullpipeline.py
  python benchmarks/bench_minicheck_fullpipeline.py --smoke   # 5 samples/type
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
NQ_PERTURBED = Path(__file__).parent / "datasets" / "nq_500_perturbed.jsonl"
SUMMEVAL = Path(__file__).parent / "datasets" / "summeval.jsonl"
CACHE = RESULTS_DIR / "minicheck_fullpipeline_cache.json"
OUT_MD = RESULTS_DIR / "minicheck_fullpipeline.md"

# Auditor configs: only groundedness_backbone differs
DEBERTA_CFG = dict(groundedness_backbone="deberta-base")
MINICHECK_CFG = dict(groundedness_backbone="minicheck-roberta-large")

# --------------------------------------------------------------------------
# Test-set builders
# --------------------------------------------------------------------------

def _load_nq(max_per_level: int | None = None) -> list[dict]:
    records: dict[int, list[dict]] = {i: [] for i in range(5)}
    with open(NQ_PERTURBED, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lvl = rec["perturbation_level"]
            if max_per_level is None or len(records[lvl]) < max_per_level:
                records[lvl].append(rec)
    out = []
    for lvl in range(5):
        out.extend(records[lvl])
    return out


def _load_summeval(n: int | None = None) -> list[dict]:
    recs = []
    with open(SUMMEVAL, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
            if n and len(recs) >= n:
                break
    return recs


def build_edge_cases() -> list[dict]:
    """Hand-built edge cases for integrity testing."""
    return [
        # no-context fallback
        {"type": "no_context", "query": "What is the capital of France?",
         "response": "Paris is the capital of France.", "context": None},
        # empty response
        {"type": "empty_response", "query": "Summarize this.", "response": " ",
         "context": ["France is a country in western Europe."]},
        # single-sentence response (consistency inapplicability)
        {"type": "single_sentence", "query": "What is photosynthesis?",
         "response": "Photosynthesis converts CO2 and water into glucose.",
         "context": ["Plants use sunlight to convert CO2 and water into glucose and oxygen."]},
        # generic query (relevance inapplicability)
        {"type": "generic_query", "query": "Summarize the following article.",
         "response": "The article discusses climate change and its effects.",
         "context": ["Climate change refers to long-term shifts in global temperatures."]},
        # multi-claim response
        {"type": "multi_claim",
         "query": "Tell me about coffee.",
         "response": ("Coffee originated in Ethiopia. "
                      "It was first cultivated in Yemen in the 15th century. "
                      "Brazil is the world's largest coffee producer. "
                      "Coffee contains caffeine, which is a stimulant."),
         "context": ["Coffee has origins in Ethiopia and Yemen. "
                     "Brazil produces more coffee than any other country. "
                     "Caffeine in coffee acts as a central nervous system stimulant."]},
        # numeric-heavy response
        {"type": "numeric",
         "query": "What are the specs of the Eiffel Tower?",
         "response": "The Eiffel Tower is 330 meters tall and weighs 10,100 tonnes.",
         "context": ["The Eiffel Tower stands 330 metres tall including its antenna "
                     "and has a total weight of approximately 10,100 tonnes."]},
        # fabricated / hallucinated response (should score low)
        {"type": "hallucinated",
         "query": "When was the Eiffel Tower built?",
         "response": "The Eiffel Tower was built in 1650 by Leonardo da Vinci.",
         "context": ["The Eiffel Tower was constructed between 1887 and 1889 "
                     "as the entrance arch to the 1889 World's Fair."]},
        # long-context (multiple chunks)
        {"type": "long_context",
         "query": "Summarize the key points.",
         "response": "Climate change causes rising sea levels and extreme weather events.",
         "context": [
             "Global temperatures have risen by approximately 1.1°C since pre-industrial times.",
             "Sea levels are rising at an accelerating rate due to melting ice sheets.",
             "Extreme weather events including hurricanes and droughts are becoming more frequent.",
             "The Paris Agreement aims to limit warming to 1.5°C above pre-industrial levels.",
             "Renewable energy adoption is critical to reducing greenhouse gas emissions.",
         ]},
    ]


def build_set_a(n_per_type: int = 10, seed: int = 42) -> list[dict]:
    """~100-input integrity set covering all pipeline paths."""
    rng = random.Random(seed)
    inputs = []

    # NQ-500: short RAG (A0 = grounded, A4 = hallucinated)
    nq = _load_nq(max_per_level=n_per_type * 4)
    for rec in rng.sample([r for r in nq if r["perturbation_level"] == 0], n_per_type):
        inputs.append({"type": "short_rag_grounded",
                       "query": rec["query"], "response": rec["response"],
                       "context": [rec["context"]]})
    for rec in rng.sample([r for r in nq if r["perturbation_level"] == 4], n_per_type):
        inputs.append({"type": "short_rag_hallucinated",
                       "query": rec["query"], "response": rec["response"],
                       "context": [rec["context"]]})

    # SummEval: long-document context
    if SUMMEVAL.exists():
        se = _load_summeval(n_per_type * 8)
        sample = rng.sample(se, min(n_per_type * 2, len(se)))
        for rec in sample:
            inputs.append({"type": "long_doc",
                           "query": "Summarize the following article.",
                           "response": rec["summary"],
                           "context": [rec["source"]]})

    # Edge cases (all included for integrity)
    for ec in build_edge_cases():
        inputs.append(ec)

    # Multi-level NQ mix
    for lvl in (1, 2, 3):
        lvl_recs = [r for r in nq if r["perturbation_level"] == lvl]
        for rec in rng.sample(lvl_recs, min(n_per_type // 2, len(lvl_recs))):
            inputs.append({"type": f"nq_level_{lvl}",
                           "query": rec["query"], "response": rec["response"],
                           "context": [rec["context"]]})

    return inputs


def build_set_b(n_per_type: int = 60, seed: int = 99) -> list[dict]:
    """~300-input latency set, stratified by input type."""
    rng = random.Random(seed)
    inputs = []

    nq = _load_nq()

    for lvl, label in [(0, "short_rag_grounded"), (4, "short_rag_hallucinated"),
                        (1, "nq_level_1"), (2, "nq_level_2"), (3, "nq_level_3")]:
        pool = [r for r in nq if r["perturbation_level"] == lvl]
        for rec in rng.sample(pool, min(n_per_type, len(pool))):
            inputs.append({"type": label, "query": rec["query"],
                           "response": rec["response"], "context": [rec["context"]]})

    if SUMMEVAL.exists():
        se = _load_summeval()
        sample = rng.sample(se, min(n_per_type, len(se)))
        for rec in sample:
            inputs.append({"type": "long_doc",
                           "query": "Summarize the following article.",
                           "response": rec["summary"], "context": [rec["source"]]})

    # No-context (fast path)
    for rec in rng.sample([r for r in nq if r["perturbation_level"] == 0],
                           min(n_per_type // 3, 20)):
        inputs.append({"type": "no_context", "query": rec["query"],
                       "response": rec["response"], "context": None})

    rng.shuffle(inputs)
    return inputs


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_one(auditor, inp: dict, suppress_warnings: bool = True) -> dict:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore")
        try:
            r = auditor.score(
                query=inp["query"],
                response=inp["response"],
                context=inp["context"],
            )
            lat = (time.perf_counter() - t0) * 1000
            return {
                "ok": True,
                "iqs": r.iqs,
                "groundedness": r.groundedness,
                "completeness": r.completeness,
                "relevance": r.relevance,
                "consistency": r.consistency,
                "confidence": r.confidence,
                "inapplicable": list(getattr(r, "inapplicable_dimensions", None) or []),
                "flags": list(r.flags or []),
                "has_evidence_map": r.evidence_map is not None,
                "evidence_entries": len(r.evidence_map.entries) if r.evidence_map else 0,
                "lat_ms": round(lat, 1),
            }
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000
            return {"ok": False, "error": str(e)[:200], "lat_ms": round(lat, 1)}


def run_set(auditor, inputs: list[dict], label: str,
            report_every: int = 25) -> list[dict]:
    results = []
    lats = []
    for i, inp in enumerate(inputs):
        res = score_one(auditor, inp)
        results.append(res)
        lats.append(res["lat_ms"])
        if (i + 1) % report_every == 0:
            print(f"  {label}: {i+1}/{len(inputs)}  "
                  f"{np.mean(lats):.0f}ms/sample", flush=True)
    return results


# --------------------------------------------------------------------------
# Integrity checks
# --------------------------------------------------------------------------

def check_integrity(set_a_inputs, deb_results, mc_results) -> dict[str, str]:
    checks: dict[str, str] = {}

    # 1. No crashes
    crashes_deb = [i for i, r in enumerate(deb_results) if not r["ok"]]
    crashes_mc = [i for i, r in enumerate(mc_results) if not r["ok"]]
    checks["no_crashes_deberta"] = (
        "PASS" if not crashes_deb
        else f"FAIL: {len(crashes_deb)} crashes at idx {crashes_deb[:5]}")
    checks["no_crashes_minicheck"] = (
        "PASS" if not crashes_mc
        else f"FAIL: {len(crashes_mc)} crashes at idx {crashes_mc[:5]}")

    # 2. Other 4 dimensions identical
    mismatches = []
    for i, (d, m) in enumerate(zip(deb_results, mc_results)):
        if not d["ok"] or not m["ok"]:
            continue
        for dim in ("completeness", "relevance", "consistency", "confidence"):
            if d[dim] != m[dim]:
                mismatches.append((i, dim, d[dim], m[dim]))
    checks["other_dims_identical"] = (
        "PASS" if not mismatches
        else f"FAIL: {len(mismatches)} mismatches: {mismatches[:3]}")

    # 3. Evidence map populates for context inputs
    ctx_mc = [(i, r) for i, (inp, r) in enumerate(zip(set_a_inputs, mc_results))
              if inp.get("context") and r["ok"]]
    ev_missing = [i for i, r in ctx_mc if not r["has_evidence_map"]]
    checks["evidence_map_present"] = (
        "PASS" if not ev_missing
        else f"FAIL: missing evidence map on {len(ev_missing)} context inputs")

    # 4. Fallback (no-context) works; groundedness should be None
    no_ctx_mc = [(i, inp, r) for i, (inp, r) in enumerate(zip(set_a_inputs, mc_results))
                 if inp.get("context") is None and r["ok"]]
    fallback_bad = [i for i, inp, r in no_ctx_mc if r["groundedness"] is not None]
    checks["no_context_fallback"] = (
        "PASS" if not fallback_bad
        else f"FAIL: groundedness not None for {len(fallback_bad)} no-context inputs")

    # 5. IQS sanity: grounded A0 inputs should score above hallucinated A4
    a0 = [r["iqs"] for inp, r in zip(set_a_inputs, mc_results)
          if inp.get("type") == "short_rag_grounded" and r["ok"]]
    a4 = [r["iqs"] for inp, r in zip(set_a_inputs, mc_results)
          if inp.get("type") == "short_rag_hallucinated" and r["ok"]]
    if a0 and a4:
        checks["iqs_sanity_a0_gt_a4"] = (
            f"PASS (A0 mean={np.mean(a0):.3f} > A4 mean={np.mean(a4):.3f})"
            if np.mean(a0) > np.mean(a4)
            else f"FAIL (A0={np.mean(a0):.3f}, A4={np.mean(a4):.3f})")
    else:
        checks["iqs_sanity_a0_gt_a4"] = "N/A (insufficient samples)"

    return checks


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def check_determinism(auditor, inputs: list[dict],
                      n_inputs: int = 20, n_passes: int = 10) -> int:
    sub = inputs[:n_inputs]
    runs: list[list[dict]] = []
    for p in range(n_passes):
        run = []
        for inp in sub:
            r = score_one(auditor, inp)
            run.append(r)
        runs.append(run)

    base = runs[0]
    deviations = 0
    for run in runs[1:]:
        for b, r in zip(base, run):
            if not b["ok"] or not r["ok"]:
                continue
            for field in ("iqs", "groundedness", "completeness",
                          "relevance", "consistency", "confidence"):
                if b[field] != r[field]:
                    deviations += 1
            if b["flags"] != r["flags"]:
                deviations += 1
    return deviations


# --------------------------------------------------------------------------
# Latency analysis
# --------------------------------------------------------------------------

def latency_stats(results: list[dict]) -> dict:
    lats = [r["lat_ms"] for r in results if r["ok"]]
    if not lats:
        return {}
    return {
        "mean": round(float(np.mean(lats)), 1),
        "p50": round(float(np.percentile(lats, 50)), 1),
        "p95": round(float(np.percentile(lats, 95)), 1),
        "n": len(lats),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_md(set_a_inputs, deb_a, mc_a, set_b_inputs, deb_b, mc_b,
             integrity, det_devs, deb_load_s, mc_load_s, args) -> None:
    lines = [
        "# MiniCheck Full-Pipeline Integration & Latency",
        "",
        f"Branch: `bench/minicheck-fullpipeline`. "
        f"Real `Auditor.score()` path, both backbones on the same inputs.",
        "",
    ]

    # --- Integrity ---
    lines += ["## Step 4 — End-to-end integrity (Set A)", ""]
    all_pass = all(v.startswith("PASS") for v in integrity.values())
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    for k, v in integrity.items():
        icon = "✓" if v.startswith("PASS") else "✗"
        lines.append(f"- {icon} `{k}`: {v}")
    lines.append("")

    # --- Determinism ---
    lines += ["## Step 5 — End-to-end determinism", ""]
    lines.append(f"20 inputs × 10 passes, MiniCheck backbone.")
    lines.append(
        f"**{det_devs} deviations** — "
        + ("PASS ✓" if det_devs == 0 else "FAIL ✗"))
    lines.append("")

    # --- Latency ---
    lines += ["## Step 6 — Latency comparison (Set B)", ""]
    lines.append(f"Model cold-start load: deberta {deb_load_s:.1f}s | "
                 f"MiniCheck {mc_load_s:.1f}s")
    lines.append("")

    type_order = ["short_rag_grounded", "short_rag_hallucinated",
                  "long_doc", "no_context",
                  "nq_level_1", "nq_level_2", "nq_level_3"]
    type_label = {
        "short_rag_grounded": "short RAG (grounded)",
        "short_rag_hallucinated": "short RAG (hallucinated)",
        "long_doc": "long document (SummEval)",
        "no_context": "no-context fallback",
        "nq_level_1": "NQ level 1",
        "nq_level_2": "NQ level 2",
        "nq_level_3": "NQ level 3",
    }
    lines += [
        "| Input type | deberta-base | MiniCheck-RoBERTa-L | slowdown |",
        "|---|---|---|---|",
    ]
    overall_deb, overall_mc = [], []
    for t in type_order:
        d_res = [r for inp, r in zip(set_b_inputs, deb_b)
                 if inp.get("type") == t and r["ok"]]
        m_res = [r for inp, r in zip(set_b_inputs, mc_b)
                 if inp.get("type") == t and r["ok"]]
        if not d_res or not m_res:
            continue
        d_lats = [r["lat_ms"] for r in d_res]
        m_lats = [r["lat_ms"] for r in m_res]
        overall_deb.extend(d_lats)
        overall_mc.extend(m_lats)
        d_mean = np.mean(d_lats)
        m_mean = np.mean(m_lats)
        slow = m_mean / d_mean if d_mean > 0 else 0
        lines.append(f"| {type_label.get(t, t)} | {d_mean:.0f}ms | {m_mean:.0f}ms "
                     f"| {slow:.2f}x |")

    if overall_deb and overall_mc:
        d_s = latency_stats([{"ok": True, "lat_ms": l} for l in overall_deb])
        m_s = latency_stats([{"ok": True, "lat_ms": l} for l in overall_mc])
        slow_mean = m_s["mean"] / d_s["mean"] if d_s["mean"] > 0 else 0
        slow_p95 = m_s["p95"] / d_s["p95"] if d_s["p95"] > 0 else 0
        lines.append(f"| **OVERALL mean** | **{d_s['mean']}ms** | **{m_s['mean']}ms** "
                     f"| **{slow_mean:.2f}x** |")
        lines.append(f"| **OVERALL p50 / p95** | {d_s['p50']}ms / {d_s['p95']}ms "
                     f"| {m_s['p50']}ms / {m_s['p95']}ms "
                     f"| {slow_mean:.2f}x mean / {slow_p95:.2f}x p95 |")
        lines.append("")
        lines.append(f"n={len(overall_deb)} inputs each. "
                     f"Warm-cache latency (models pre-loaded).")

    # --- IQS behavior ---
    lines += ["", "## Step 7 — Composite IQS behavior", ""]
    valid = [(inp, d, m) for inp, d, m in zip(set_a_inputs, deb_a, mc_a)
             if d["ok"] and m["ok"]]
    if valid:
        d_iqs = [d["iqs"] for _, d, _ in valid]
        m_iqs = [m["iqs"] for _, _, m in valid]
        d_gnd = [d["groundedness"] for _, d, _ in valid if d["groundedness"] is not None]
        m_gnd = [m["groundedness"] for _, _, m in valid if m["groundedness"] is not None]
        lines.append(f"Mean composite IQS: deberta={np.mean(d_iqs):.3f} | "
                     f"MiniCheck={np.mean(m_iqs):.3f}")
        if d_gnd and m_gnd:
            lines.append(f"Mean groundedness: deberta={np.mean(d_gnd):.3f} | "
                         f"MiniCheck={np.mean(m_gnd):.3f}")
        collapse_deb = sum(
            1 for inp2, d2, _ in valid
            if inp2.get("type") == "short_rag_grounded" and d2["iqs"] < 0.01)
        collapse_mc = sum(
            1 for inp2, _, m2 in valid
            if inp2.get("type") == "short_rag_grounded" and m2["iqs"] < 0.01)
        lines.append(f"Collapse-to-zero on grounded inputs: "
                     f"deberta={collapse_deb} | MiniCheck={collapse_mc}")
    lines.append("")

    # Spot-check table
    lines += ["**Spot-check (5 inputs):**", "",
              "| Type | deberta IQS / gnd | MiniCheck IQS / gnd |",
              "|---|---|---|"]
    shown = 0
    for inp, d, m in valid:
        if shown >= 5:
            break
        if not inp.get("type"):
            continue
        d_g = f"{d['groundedness']:.3f}" if d["groundedness"] is not None else "None"
        m_g = f"{m['groundedness']:.3f}" if m["groundedness"] is not None else "None"
        lines.append(f"| {inp['type']} | {d['iqs']:.3f} / {d_g} "
                     f"| {m['iqs']:.3f} / {m_g} |")
        shown += 1
    lines.append("")

    # --- Recommendation ---
    lines += ["## Recommendation", ""]
    if overall_deb and overall_mc:
        slow = m_s["mean"] / d_s["mean"]
        if slow < 1.5:
            tier = "strong case for DEFAULT"
            rec = (f"Full-pipeline slowdown is {slow:.2f}x (< 1.5x threshold). "
                   "Groundedness is only one of five dimensions; the pipeline "
                   "overhead amortises the backbone cost. Near-perfect NQ-500 "
                   "AUC (0.991) + improved SummEval correlation (+0.04 rho) "
                   "at modest latency cost → **strong case to promote MiniCheck "
                   "to the default backbone**. Human decision required.")
        elif slow < 2.0:
            tier = "opt-in high_accuracy"
            rec = (f"Full-pipeline slowdown is {slow:.2f}x (1.5–2.0x). "
                   "Meaningful latency cost but not prohibitive. "
                   "Recommend shipping as `high_accuracy` opt-in option. "
                   "Human decision on whether the correlation and discrimination "
                   "gains justify the speed cost.")
        else:
            tier = "opt-in only — latency cost too high for default"
            rec = (f"Full-pipeline slowdown is {slow:.2f}x (> 2x). "
                   "Keep deberta-base as default; ship MiniCheck as "
                   "`high_accuracy` opt-in for latency-tolerant use cases.")
        lines.append(f"**Tier: {tier}**")
        lines.append("")
        lines.append(rec)
    lines.append("")
    lines.append("Summary:")
    lines.append(f"- SummEval correlation: rho 0.4251 → 0.4659 (+0.04, deberta vs MiniCheck)")
    lines.append(f"- NQ-500 AUC: 0.8748 → 0.9910 (+0.116, deberta vs MiniCheck)")
    lines.append(f"- Determinism: {det_devs} deviations (required: 0)")
    lines.append(f"- Integrity: {'all checks PASS' if all_pass else 'see failures above'}")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 5 samples per type")
    ap.add_argument("--skip-set-b", action="store_true",
                    help="Skip latency Set B (integrity + determinism only)")
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()

    n_per_type = 5 if args.smoke else 10
    n_per_type_b = 5 if args.smoke else 60

    print("Building test sets...", flush=True)
    set_a = build_set_a(n_per_type=n_per_type)
    set_b = [] if args.skip_set_b else build_set_b(n_per_type=n_per_type_b)
    print(f"Set A: {len(set_a)} inputs | Set B: {len(set_b)} inputs", flush=True)

    from scroot import Auditor

    print("\n--- Loading deberta-base ---", flush=True)
    t0 = time.perf_counter()
    aud_deb = Auditor(**DEBERTA_CFG)
    # warm up model cache
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aud_deb.score("warmup", "warmup", context=["warmup"])
    deb_load_s = time.perf_counter() - t0
    print(f"  loaded in {deb_load_s:.1f}s", flush=True)

    print("\n--- Loading MiniCheck-RoBERTa-Large ---", flush=True)
    t0 = time.perf_counter()
    aud_mc = Auditor(**MINICHECK_CFG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aud_mc.score("warmup", "warmup", context=["warmup"])
    mc_load_s = time.perf_counter() - t0
    print(f"  loaded in {mc_load_s:.1f}s", flush=True)

    # --- Set A: integrity ---
    print(f"\n--- Set A: deberta-base ({len(set_a)} inputs) ---", flush=True)
    deb_a = run_set(aud_deb, set_a, "deberta-base")
    print(f"\n--- Set A: minicheck ({len(set_a)} inputs) ---", flush=True)
    mc_a = run_set(aud_mc, set_a, "minicheck")

    print("\n--- Integrity checks ---", flush=True)
    integrity = check_integrity(set_a, deb_a, mc_a)
    for k, v in integrity.items():
        icon = "✓" if v.startswith("PASS") else "✗"
        print(f"  {icon} {k}: {v}", flush=True)

    # --- Determinism ---
    det_devs = 0
    if not args.skip_determinism:
        n_det = min(5 if args.smoke else 20, len(set_a))
        print(f"\n--- Determinism: {n_det} inputs × 10 passes ---", flush=True)
        det_devs = check_determinism(aud_mc, set_a, n_inputs=n_det)
        print(f"  Deviations: {det_devs} ({'PASS' if det_devs == 0 else 'FAIL'})",
              flush=True)

    # --- Set B: latency ---
    deb_b, mc_b = [], []
    if set_b:
        print(f"\n--- Set B latency: deberta-base ({len(set_b)} inputs) ---", flush=True)
        deb_b = run_set(aud_deb, set_b, "deberta-base", report_every=50)
        print(f"\n--- Set B latency: minicheck ({len(set_b)} inputs) ---", flush=True)
        mc_b = run_set(aud_mc, set_b, "minicheck", report_every=50)

        d_stats = latency_stats(deb_b)
        m_stats = latency_stats(mc_b)
        slow = m_stats["mean"] / d_stats["mean"] if d_stats["mean"] > 0 else 0
        print(f"\nLatency summary (n={d_stats['n']}):")
        print(f"  deberta: mean={d_stats['mean']}ms p50={d_stats['p50']}ms "
              f"p95={d_stats['p95']}ms")
        print(f"  minicheck: mean={m_stats['mean']}ms p50={m_stats['p50']}ms "
              f"p95={m_stats['p95']}ms")
        print(f"  slowdown: {slow:.2f}x mean", flush=True)

    write_md(set_a, deb_a, mc_a, set_b, deb_b, mc_b,
             integrity, det_devs, deb_load_s, mc_load_s, args)


if __name__ == "__main__":
    main()
