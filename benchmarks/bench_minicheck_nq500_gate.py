"""NQ-500 discrimination gate for MiniCheck-RoBERTa-Large backbone.

Scores all 2,500 NQ-500 perturbed records (A0-A4) with BOTH:
  - deberta-base (current default, reference)
  - MiniCheck-RoBERTa-Large (candidate opt-in backbone)

using the SAME groundedness harness as Experiment A (top-8 premise
retrieval, coverage-ratio aggregation). Metric functions imported from
bench_correlation to guarantee identical methodology.

Gate: MiniCheck binary AUC (A0 vs A4) must be >= 0.85.

Run:
  $env:PYTHONIOENCODING="utf-8"
  python benchmarks/bench_minicheck_nq500_gate.py
  python benchmarks/bench_minicheck_nq500_gate.py --max 20   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add src/ so scroot is importable. bench_gap_backbone_ab's own module-level
# code removes benchmarks/ from sys.path when it is imported (preventing the
# local datasets/ dir from shadowing the HF datasets package), so we don't
# need to do that here; just ensure src/ is present before the import.
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np  # noqa: E402

from bench_gap_backbone_ab import (  # noqa: E402
    DebertaBackbone, MiniCheckRoberta,
    ENTAIL_THRESHOLD, TOP_K_PREMISES, EMB_MODEL,
    chunk_article, context_sentences,
)

DATASET = Path(__file__).parent / "datasets" / "nq_500_perturbed.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
CACHE = RESULTS_DIR / "minicheck_nq500_gate_scores.json"
OUT_MD = RESULTS_DIR / "minicheck_nq500_gate.md"
GAP_REPORT = Path(__file__).parent.parent / "GAP_CLOSING_REPORT.md"

AUC_GATE = 0.85
DEBERTA_REFERENCE_AUC = 0.8625   # IQS AUC from prior bench_correlation run


def _binary_auc(pos: list[float], neg: list[float]) -> float:
    """Wilcoxon-Mann-Whitney AUC: P(pos > neg). Identical to bench_correlation."""
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    total = len(pos) * len(neg)
    return (wins + 0.5 * ties) / total if total else 0.5


def _kendall_tau(x: list[float], y: list[float]) -> float:
    from scipy.stats import kendalltau
    tau, _ = kendalltau(x, y)
    return float(tau)


def load_nq500(max_examples: int | None = None) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ex_id = rec["id"]
            if max_examples is not None:
                if ex_id not in seen and len(seen) >= max_examples:
                    continue
                seen.add(ex_id)
            records.append(rec)
    return records


def score_backbone(backbone, records: list[dict], emb_model,
                   label: str) -> list[dict]:
    """Score each NQ-500 record with a groundedness backbone.

    Returns list of {id, perturbation_level, groundedness} dicts.
    """
    from scroot.text_utils import extract_atomic_claims

    per_record: list[dict] = []
    lats: list[float] = []

    for n, rec in enumerate(records):
        t0 = time.perf_counter()
        claims = extract_atomic_claims(rec["response"])
        ctx = context_sentences(chunk_article(rec["context"]))

        if not claims or not ctx:
            per_record.append({
                "id": rec["id"],
                "perturbation_level": rec["perturbation_level"],
                "groundedness": 1.0 if not claims else 0.0,
                "claim_scores": [],
            })
            lats.append((time.perf_counter() - t0) * 1000)
            continue

        ctx_emb = emb_model.encode(ctx, convert_to_numpy=True)
        claim_emb = emb_model.encode(claims, convert_to_numpy=True)
        cn = np.linalg.norm(ctx_emb, axis=1) + 1e-8

        claim_scores: list[float] = []
        for ci, claim in enumerate(claims):
            v = claim_emb[ci]
            sims = ctx_emb @ v / (cn * (np.linalg.norm(v) + 1e-8))
            k = min(TOP_K_PREMISES, len(ctx))
            top = np.argsort(sims)[::-1][:k]
            pairs = [(ctx[j], claim) for j in top]
            probs = backbone.score_pairs(pairs)
            claim_scores.append(float(max(probs)) if probs else 0.0)

        grounded = sum(1 for s in claim_scores if s >= ENTAIL_THRESHOLD)
        gs = grounded / len(claims)
        per_record.append({
            "id": rec["id"],
            "perturbation_level": rec["perturbation_level"],
            "groundedness": round(gs, 6),
            "claim_scores": [round(s, 4) for s in claim_scores],
        })
        lats.append((time.perf_counter() - t0) * 1000)

        if (n + 1) % 250 == 0:
            print(f"    {label}: {n+1}/{len(records)}  "
                  f"{np.mean(lats):.0f}ms/sample", flush=True)

    return per_record


def compute_metrics(per_record: list[dict]) -> dict:
    level_scores: dict[int, list[float]] = {lvl: [] for lvl in range(5)}
    for r in per_record:
        level_scores[r["perturbation_level"]].append(r["groundedness"])

    levels = [r["perturbation_level"] for r in per_record]
    gs_vals = [r["groundedness"] for r in per_record]

    from scipy.stats import spearmanr
    rho = float(spearmanr(levels, gs_vals).correlation)
    tau = _kendall_tau(levels, gs_vals)

    a0, a3, a4 = level_scores[0], level_scores[3], level_scores[4]
    auc_a0_vs_a4 = _binary_auc(a0, a4)
    auc_a0_vs_a3 = _binary_auc(a0, a3)

    threshold = 0.5
    bin_acc = (sum(1 for s in a0 if s >= threshold)
               + sum(1 for s in a4 if s < threshold)) / (len(a0) + len(a4))

    per_level = {}
    for lvl in range(5):
        sub = level_scores[lvl]
        mean = float(np.mean(sub)) if sub else 0.0
        per_level[f"A{lvl}"] = round(mean, 4)

    means = [per_level[f"A{i}"] for i in range(5)]
    mean_sep = round(means[0] - means[4], 4)

    return {
        "auc_a0_vs_a4": round(auc_a0_vs_a4, 4),
        "auc_a0_vs_a3": round(auc_a0_vs_a3, 4),
        "spearman_rho": round(rho, 4),
        "kendall_tau": round(tau, 4),
        "binary_accuracy": round(bin_acc, 4),
        "mean_separation": mean_sep,
        "per_level_means": per_level,
    }


def determinism_check(backbone, records: list[dict], emb_model,
                      n_examples: int = 10, repeats: int = 10) -> int:
    sub = records[:n_examples]
    runs = []
    for _ in range(repeats):
        res = score_backbone(backbone, sub, emb_model, "det_check")
        runs.append([round(r["groundedness"], 6) for r in res])
    base = runs[0]
    return sum(1 for r in runs for a, b in zip(base, r) if a != b)


def write_md(results: dict, verdict: str, args) -> None:
    deb = results["deberta-base"]
    mc = results["minicheck-roberta-large"]
    gate_deb = "PASS" if deb["auc_a0_vs_a4"] >= AUC_GATE else "FAIL"
    gate_mc = "**PASS**" if mc["auc_a0_vs_a4"] >= AUC_GATE else (
        "MARGINAL" if mc["auc_a0_vs_a4"] >= 0.82 else "**FAIL**")

    lines = [
        "# MiniCheck-RoBERTa-Large NQ-500 Discrimination Gate",
        "",
        f"2,500 NQ-500 perturbed records (A0–A4). Same harness as Exp A "
        f"(top-{TOP_K_PREMISES} premise retrieval, coverage-ratio aggregation"
        + (f", --max {args.max} examples" if args.max else "") + ").",
        "",
        "## Side-by-side metric table",
        "",
        "| Metric | deberta-base (default) | MiniCheck-RoBERTa-L (candidate) | Gate |",
        "|---|---|---|---|",
        f"| **AUC (A0 vs A4)** | {deb['auc_a0_vs_a4']} ({gate_deb}) "
        f"| {mc['auc_a0_vs_a4']} ({gate_mc}) | ≥ {AUC_GATE} |",
        f"| AUC (A0 vs A3) | {deb['auc_a0_vs_a3']} "
        f"| {mc['auc_a0_vs_a3']} | — |",
        f"| Spearman ρ (gnd vs level) | {deb['spearman_rho']} "
        f"| {mc['spearman_rho']} | — |",
        f"| Kendall τ | {deb['kendall_tau']} "
        f"| {mc['kendall_tau']} | — |",
        f"| Binary accuracy (thr=0.5) | {deb['binary_accuracy']} "
        f"| {mc['binary_accuracy']} | — |",
        f"| Mean separation (A0−A4) | {deb['mean_separation']} "
        f"| {mc['mean_separation']} | — |",
        "",
        "## Per-level mean groundedness",
        "",
        "| Level | deberta-base | MiniCheck-RoBERTa-L | Monotone? |",
        "|---|---|---|---|",
    ]
    prev_deb = prev_mc = None
    for lvl in range(5):
        key = f"A{lvl}"
        d = deb["per_level_means"][key]
        m = mc["per_level_means"][key]
        deb_ok = "↓" if prev_deb is not None and d < prev_deb else (
            "—" if prev_deb is None else "⚠ UP")
        mc_ok = "↓" if prev_mc is not None and m < prev_mc else (
            "—" if prev_mc is None else "⚠ UP")
        lines.append(f"| {key} | {d} ({deb_ok}) | {m} ({mc_ok}) | |")
        prev_deb, prev_mc = d, m

    det_deb = results.get("det_deviations_deberta", "n/a")
    det_mc = results.get("det_deviations_minicheck", "n/a")
    lines += [
        "",
        "## Determinism check (10 examples × 10 passes)",
        "",
        f"- deberta-base: {det_deb} deviations",
        f"- MiniCheck-RoBERTa-Large: {det_mc} deviations",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]

    auc = mc["auc_a0_vs_a4"]
    if auc >= AUC_GATE:
        lines += [
            f"MiniCheck-RoBERTa-Large AUC = {auc} ≥ {AUC_GATE}. Gate PASSED.",
            "Hallucination discrimination is preserved. Safe to ship as opt-in "
            "`high_accuracy` backbone.",
            "",
            f"SummEval correlation gain: rho 0.4251 → 0.4659 (+0.04) vs deberta-base.",
            f"NQ-500 AUC: {deb['auc_a0_vs_a4']} (deberta) → {auc} (MiniCheck).",
        ]
    elif auc >= 0.82:
        lines += [
            f"MiniCheck-RoBERTa-Large AUC = {auc}, between 0.82 and {AUC_GATE}.",
            "Below the hard gate but above the marginal floor. Human decision required.",
            "Correlation gain (+0.04 SummEval rho) vs. discrimination cost ("
            f"-{round(deb['auc_a0_vs_a4'] - auc, 4)} AUC) — not an auto-adopt.",
        ]
    else:
        lines += [
            f"MiniCheck-RoBERTa-Large AUC = {auc} < 0.82. Gate FAILED.",
            "Do NOT ship MiniCheck as an option. Better summary correlation "
            "is not worth weaker fabrication detection in a grounding tool.",
            "",
            "**Paper framing:** Purpose-built factuality models (MiniCheck) "
            "improved human-rated summary correlation (+0.04 rho on SummEval) "
            "but degraded adversarial fabrication discrimination "
            f"(AUC {deb['auc_a0_vs_a4']} → {auc} on NQ-500). This illustrates "
            "a tension between the two evaluation regimes: SummEval measures "
            "overall summary quality alignment, while NQ-500 measures binary "
            "hallucination detection. A backbone optimised for one may not "
            "preserve performance on the other.",
        ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None,
                    help="Limit to first N distinct NQ examples (smoke test)")
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()

    records = load_nq500(args.max)
    print(f"Loaded {len(records)} records "
          f"({len({r['id'] for r in records})} distinct examples).", flush=True)

    from scroot.models import get_embedding_model
    emb_model = get_embedding_model(EMB_MODEL, device="cpu")

    cache: dict = {}
    if CACHE.exists():
        cache = json.load(open(CACHE, encoding="utf-8"))

    backbones = [
        ("deberta-base", DebertaBackbone),
        ("minicheck-roberta-large", MiniCheckRoberta),
    ]

    scored: dict[str, list[dict]] = {}
    for bname, BClass in backbones:
        if bname in cache:
            print(f"\n=== {bname} (from cache) ===", flush=True)
            scored[bname] = cache[bname]["per_record"]
            continue
        print(f"\n=== {bname} ===", flush=True)
        t0 = time.perf_counter()
        backbone = BClass()
        print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)
        per_record = score_backbone(backbone, records, emb_model, bname)
        scored[bname] = per_record
        cache[bname] = {"per_record": per_record}
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), indent=2)
        print(f"  done. {len(per_record)} records.", flush=True)

    results: dict = {}
    for bname in ("deberta-base", "minicheck-roberta-large"):
        metrics = compute_metrics(scored[bname])
        results[bname] = metrics
        print(f"\n{bname}:")
        print(f"  AUC A0/A4:  {metrics['auc_a0_vs_a4']}")
        print(f"  AUC A0/A3:  {metrics['auc_a0_vs_a3']}")
        print(f"  Spearman ρ: {metrics['spearman_rho']}")
        print(f"  Kendall τ:  {metrics['kendall_tau']}")
        print(f"  Binary acc: {metrics['binary_accuracy']}")
        print(f"  Mean sep:   {metrics['mean_separation']}")
        print(f"  Per-level:  "
              + "  ".join(f"A{i}={metrics['per_level_means'][f'A{i}']}"
                          for i in range(5)))

    if not args.skip_determinism:
        print("\n--- Determinism checks ---", flush=True)
        det_keys = {
            "deberta-base": "det_deviations_deberta",
            "minicheck-roberta-large": "det_deviations_minicheck",
        }
        for bname, BClass in backbones:
            backbone = BClass()
            dev = determinism_check(backbone, records, emb_model)
            results[det_keys[bname]] = dev
            print(f"  {bname}: {dev} deviations", flush=True)

    mc_auc = results["minicheck-roberta-large"]["auc_a0_vs_a4"]
    if mc_auc >= AUC_GATE:
        verdict = "PASS — MiniCheck-RoBERTa-Large ships as opt-in high_accuracy backbone"
    elif mc_auc >= 0.82:
        verdict = "MARGINAL — AUC below gate, flag for human decision"
    else:
        verdict = "FAIL — MiniCheck-RoBERTa-Large does NOT ship (discrimination loss)"

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    write_md(results, verdict, args)


if __name__ == "__main__":
    main()
