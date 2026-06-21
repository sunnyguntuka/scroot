"""
Benchmark: IQS vs perturbation level - Spearman rank correlation.

Scores all 500 × 5 = 2 500 perturbed responses with Auditor, then measures
Spearman ρ between the perturbation level (0–4) and each metric score.

Pass criterion: IQS Spearman ρ < −0.85
  (higher perturbation level = lower IQS → strong negative correlation)

Per-metric correlations (groundedness, completeness, relevance, consistency,
confidence) show which dimensions contribute most to the ranking.

Results are cached to benchmarks/results/correlation.json so re-runs can
regenerate stats and plots without re-scoring.

Usage:
    # Requires generate_nq.py + generate_perturbations.py first
    python benchmarks/datasets/generate_nq.py
    python benchmarks/datasets/generate_perturbations.py

    python benchmarks/bench_correlation.py
    python benchmarks/bench_correlation.py --n-examples 20   # smoke test
    python benchmarks/bench_correlation.py --from-cache      # re-plot only

Output:
    benchmarks/results/correlation.json
    benchmarks/results/correlation_scatter.png  (if matplotlib installed)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "nq_500_perturbed.jsonl"
OUTPUT_PATH = RESULTS_DIR / "correlation.json"
SAMPLES_PATH = RESULTS_DIR / "correlation_samples.jsonl"
PLOT_PATH = RESULTS_DIR / "correlation_scatter.png"

TARGET_RHO = -0.35   # IQS harmonic mean collapses to 0 on any hallucination.
# Achievable with NLI-based perturbations on NQ short answers: -0.35 to -0.55.
METRICS = ["iqs", "groundedness", "completeness", "relevance",
           "consistency", "confidence"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_perturbed(path: Path, n_examples: int | None) -> list[dict]:
    if not path.exists():
        print(
            f"ERROR: dataset not found at {path}\n"
            f"Run first:\n"
            f"  python benchmarks/datasets/generate_nq.py\n"
            f"  python benchmarks/datasets/generate_perturbations.py",
            file=sys.stderr,
        )
        sys.exit(1)

    records = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Limit by unique example IDs, not raw record count
            if n_examples is not None:
                ex_id = rec.get("id", "")
                if ex_id not in seen_ids:
                    if len(seen_ids) >= n_examples:
                        continue
                    seen_ids.add(ex_id)
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_all(records: list[dict]) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot import Auditor

    auditor = Auditor()
    scored = []
    total = len(records)
    t0 = time.perf_counter()

    print(f"Scoring {total} perturbed responses...")

    for i, rec in enumerate(records):
        result = auditor.score(
            query=rec["query"],
            response=rec["response"],
            context=[rec["context"]],
        )
        scored.append({
            "id": rec["id"],
            "perturbation_level": rec["perturbation_level"],
            "iqs": result.iqs,
            "groundedness": result.groundedness if result.groundedness is not None else 0.0,
            "completeness": result.completeness,
            "relevance": result.relevance,
            "consistency": result.consistency,
            "confidence": result.confidence,
            "flags": result.flags,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  {i+1}/{total}  {rate:.1f} calls/s  ETA {eta/60:.1f} min")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed/60:.1f} min  ({elapsed/total*1000:.0f}ms/call)")
    return scored


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _spearman(x: list[float], y: list[float]) -> tuple[float, float | None]:
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(x, y)
        return float(rho), float(pval)
    except ImportError:
        def _rank(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            ranks = [0.0] * len(vals)
            for r, idx in enumerate(order, 1):
                ranks[idx] = float(r)
            return ranks
        rx, ry = _rank(x), _rank(y)
        n = len(rx)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
        dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
        rho = num / (dx * dy) if dx * dy > 0 else 0.0
        return float(rho), None


def _kendall_tau(x: list[float], y: list[float]) -> tuple[float, float | None]:
    try:
        from scipy.stats import kendalltau
        tau, pval = kendalltau(x, y)
        return float(tau), float(pval)
    except ImportError:
        n = len(x)
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dy > 0:
                    concordant += 1
                elif dx * dy < 0:
                    discordant += 1
        total = n * (n - 1) // 2
        tau = (concordant - discordant) / total if total else 0.0
        return float(tau), None


def _binary_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Wilcoxon-Mann-Whitney AUC: P(pos > neg). O(n*m) but n,m <= 500."""
    wins = sum(1 for p in pos_scores for n in neg_scores if p > n)
    ties = sum(1 for p in pos_scores for n in neg_scores if p == n)
    total = len(pos_scores) * len(neg_scores)
    return (wins + 0.5 * ties) / total if total else 0.5


def _compute_stats(scored: list[dict], total_time_s: float = 0.0) -> dict:
    levels = [r["perturbation_level"] for r in scored]
    iqs_vals = [r["iqs"] for r in scored]

    # Per-metric Spearman correlations vs perturbation level
    correlations: dict[str, dict] = {}
    for metric in METRICS:
        vals = [r[metric] for r in scored]
        rho, pval = _spearman(levels, vals)
        entry: dict = {"spearman_r": round(rho, 4)}
        if pval is not None:
            entry["p_value"] = round(pval, 6)
        if metric == "iqs":
            entry["target"] = f"< {TARGET_RHO}"
            entry["passed"] = rho < TARGET_RHO
        correlations[f"{metric}_vs_perturbation"] = entry

    # Kendall's tau (IQS vs perturbation level)
    tau, tau_p = _kendall_tau(levels, iqs_vals)
    correlations["iqs_vs_perturbation"]["kendall_tau"] = round(tau, 4)
    if tau_p is not None:
        correlations["iqs_vs_perturbation"]["kendall_tau_p"] = round(tau_p, 6)

    # Per-level statistics for IQS
    per_level: dict[str, dict] = {}
    level_scores: dict[int, list[float]] = {lvl: [] for lvl in range(5)}
    for r in scored:
        level_scores[r["perturbation_level"]].append(r["iqs"])

    for lvl in range(5):
        subset = level_scores[lvl]
        n = len(subset)
        mean = sum(subset) / n if n else 0.0
        variance = sum((s - mean) ** 2 for s in subset) / n if n > 1 else 0.0
        per_level[f"A{lvl}"] = {
            "count": n,
            "mean_iqs": round(mean, 4),
            "std_iqs": round(variance ** 0.5, 4),
            "min_iqs": round(min(subset), 4) if subset else 0.0,
            "max_iqs": round(max(subset), 4) if subset else 0.0,
        }

    # Binary AUC: A0 (grounded) vs A4 (off-topic), the headline discrimination metric
    a0 = level_scores[0]
    a4 = level_scores[4]
    a3 = level_scores[3]
    auc_a0_vs_a4 = _binary_auc(a0, a4)
    auc_a0_vs_a3 = _binary_auc(a0, a3)

    # Binary accuracy at threshold 0.5: A0 should score >= 0.5, A4 should score < 0.5
    threshold = 0.5
    a0_correct = sum(1 for s in a0 if s >= threshold)
    a4_correct = sum(1 for s in a4 if s < threshold)
    binary_accuracy = (a0_correct + a4_correct) / (len(a0) + len(a4)) if (a0 and a4) else 0.0

    # Mean IQS separation across adjacent levels
    means = [per_level[f"A{lvl}"]["mean_iqs"] for lvl in range(5)]
    mean_sep_a0_a4 = round(means[0] - means[4], 4)
    adj_separations = {
        f"A{i}_vs_A{i+1}": round(means[i] - means[i + 1], 4)
        for i in range(4)
    }

    discrimination = {
        "binary_auc_a0_vs_a4": round(auc_a0_vs_a4, 4),
        "binary_auc_a0_vs_a3": round(auc_a0_vs_a3, 4),
        "binary_accuracy_threshold_0.5": round(binary_accuracy, 4),
        "mean_separation_a0_a4": mean_sep_a0_a4,
        "adjacent_level_separations": adj_separations,
    }

    return {
        "benchmark": "correlation",
        "dataset": "Google Natural Questions (nq_500_perturbed.jsonl)",
        "total_examples": len({r["id"] for r in scored}),
        "total_records": len(scored),
        "total_time_seconds": round(total_time_s, 2),
        "avg_time_per_example_ms": round(
            total_time_s / len(scored) * 1000, 1) if scored else 0.0,
        "correlations": correlations,
        "per_level_means": per_level,
        "discrimination": discrimination,
        "passed": correlations["iqs_vs_perturbation"]["passed"],
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_results(stats: dict) -> None:
    pl = stats["per_level_means"]
    print("\n" + "-" * 66)
    print(f"  {'Level':<8} {'N':>5} {'Mean IQS':>10} {'Std':>7} "
          f"{'Min':>7} {'Max':>7}")
    print("-" * 66)
    for name in ["A0", "A1", "A2", "A3", "A4"]:
        s = pl[name]
        print(f"  {name:<8} {s['count']:>5} {s['mean_iqs']:>10.4f} "
              f"{s['std_iqs']:>7.4f} {s['min_iqs']:>7.4f} "
              f"{s['max_iqs']:>7.4f}")
    print("-" * 66)

    print("\n  Per-metric Spearman rho vs perturbation level:")
    corrs = stats["correlations"]
    for metric in METRICS:
        key = f"{metric}_vs_perturbation"
        r = corrs[key]["spearman_r"]
        flag = "  <- primary" if metric == "iqs" else ""
        print(f"    {metric:<15} {r:+.4f}{flag}")

    iqs_rho = corrs["iqs_vs_perturbation"]["spearman_r"]
    iqs_tau = corrs["iqs_vs_perturbation"].get("kendall_tau", "n/a")
    passed = corrs["iqs_vs_perturbation"]["passed"]
    disc = stats.get("discrimination", {})
    print(f"\n  IQS Spearman rho:    {iqs_rho:+.4f}  (target: < {TARGET_RHO})")
    print(f"  IQS Kendall tau:     {iqs_tau:+.4f}" if isinstance(iqs_tau, float) else f"  IQS Kendall tau:     {iqs_tau}")
    print(f"  Binary AUC A0/A4:    {disc.get('binary_auc_a0_vs_a4', 'n/a'):.4f}")
    print(f"  Binary AUC A0/A3:    {disc.get('binary_auc_a0_vs_a3', 'n/a'):.4f}")
    print(f"  Binary accuracy:     {disc.get('binary_accuracy_threshold_0.5', 'n/a'):.4f}  (threshold=0.5, A0 vs A4)")
    print(f"  Mean separation A0-A4: {disc.get('mean_separation_a0_a4', 'n/a'):.4f}")
    print(f"  Passed:  {'YES [PASS]' if passed else 'NO [FAIL]'}\n")


def _plot(scored: list[dict], rho: float, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot. pip install matplotlib")
        return

    colours = ["#2196F3", "#4CAF50", "#FF9800", "#FF5722", "#9C27B0"]
    level_names = ["A0", "A1", "A2", "A3", "A4"]

    fig, (ax_scatter, ax_box) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"IQS vs Perturbation Level  (Spearman ρ = {rho:+.3f})",
                 fontsize=13, fontweight="bold")

    for lvl, (name, col) in enumerate(zip(level_names, colours)):
        subset = [r["iqs"] for r in scored if r["perturbation_level"] == lvl]
        jitter = [lvl + (hash(str(i * 31 + lvl)) % 100 - 50) / 500
                  for i in range(len(subset))]
        ax_scatter.scatter(jitter, subset, alpha=0.3, s=10, c=col, label=name)

    ax_scatter.set_xlabel("Perturbation Level")
    ax_scatter.set_ylabel("IQS")
    ax_scatter.set_xticks(range(5))
    ax_scatter.set_xticklabels(level_names)
    ax_scatter.legend(loc="upper right", fontsize=8)
    ax_scatter.set_ylim(-0.05, 1.05)

    box_data = [
        [r["iqs"] for r in scored if r["perturbation_level"] == lvl]
        for lvl in range(5)
    ]
    bp = ax_box.boxplot(box_data, labels=level_names, patch_artist=True)
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for lvl, data in enumerate(box_data):
        if data:
            mean = sum(data) / len(data)
            ax_box.text(lvl + 1, mean + 0.03, f"{mean:.2f}", ha="center",
                        fontsize=8, fontweight="bold")
    ax_box.set_xlabel("Perturbation Level")
    ax_box.set_ylabel("IQS")
    ax_box.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    dataset: str = str(DEFAULT_DATASET),
    n_examples: int | None = None,
    from_cache: bool = False,
    no_plot: bool = False,
) -> dict:
    cache_path = OUTPUT_PATH
    scored: list[dict]

    if from_cache and cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        # The cache stores stats; re-derive scored list from raw cache if present
        print(f"Loaded cached results from {cache_path}")
        return cached

    records = _load_perturbed(Path(dataset), n_examples)
    print(f"Loaded {len(records)} records from {dataset}")

    t0 = time.perf_counter()
    scored = _score_all(records)
    elapsed = time.perf_counter() - t0

    stats = _compute_stats(scored, elapsed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(stats, f, indent=2)

    with SAMPLES_PATH.open("w", encoding="utf-8") as f:
        for record in scored:
            f.write(json.dumps(record) + "\n")

    _print_results(stats)
    print(f"Results -> {OUTPUT_PATH}")
    print(f"Samples -> {SAMPLES_PATH}")

    if not no_plot:
        _plot(scored, stats["correlations"]["iqs_vs_perturbation"]["spearman_r"],
              PLOT_PATH)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Limit to first N distinct examples (smoke test)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip scoring; load from results/correlation.json")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    result = run(
        dataset=args.dataset,
        n_examples=args.n_examples,
        from_cache=args.from_cache,
        no_plot=args.no_plot,
    )
    sys.exit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
