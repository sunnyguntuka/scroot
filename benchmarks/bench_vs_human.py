"""
Benchmark: IQS vs human quality judgments - Pearson and Spearman correlation.

Primary source: benchmarks/datasets/human_judgments_50.jsonl
  A committed file of 50 hand-labeled (query, response, context, human_score)
  examples where human_score is a 1–5 Likert quality rating.

Fallback source: SummEval dataset (HuggingFace mteb/summeval)
  Used if the committed file is not found; contains ~100 machine summaries
  with human consistency annotations (1–5, 3 annotators).

Metrics computed:
  - Pearson r  (IQS vs human score)   ← primary pass/fail
  - Spearman ρ (IQS vs human score)
  - MAE (mean absolute error, after normalising human 1–5 → 0–1)
  - Groundedness vs human score (Pearson)

Pass criterion: Pearson r > 0.80

Usage:
    python benchmarks/bench_vs_human.py
    python benchmarks/bench_vs_human.py --from-cache
    python benchmarks/bench_vs_human.py --use-summeval   # force SummEval

Output:
    benchmarks/results/human_correlation.json
    benchmarks/results/human_scatter.png  (if matplotlib installed)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
COMMITTED_FILE = Path(__file__).parent / "datasets" / "human_judgments_50.jsonl"
OUTPUT_PATH = RESULTS_DIR / "human_correlation.json"
PLOT_PATH = RESULTS_DIR / "human_scatter.png"

TARGET_PEARSON = 0.80


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_committed(path: Path) -> list[dict]:
    """Load the hand-labeled human_judgments_50.jsonl."""
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items.append({
                "query": rec["query"],
                "response": rec["response"],
                "context": rec.get("context"),
                "human_score": float(rec["human_score"]),
                "source": "human_judgments_50",
            })
    print(f"Loaded {len(items)} examples from committed file {path}")
    return items


def _load_summeval(n: int | None) -> list[dict]:
    """Load SummEval from HuggingFace as fallback."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Loading SummEval from HuggingFace (fallback)...")
    try:
        ds = load_dataset("mteb/summeval", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"ERROR loading SummEval: {e}", file=sys.stderr)
        sys.exit(1)

    items = []
    for row in ds:
        article = row.get("text", "")
        summaries = row.get("machine_summaries", [])
        annotations = row.get("human_annotations", [])
        for summary, ann in zip(summaries, annotations):
            if not summary or not article:
                continue
            consistency = ann.get("consistency")
            if consistency is None:
                continue
            if isinstance(consistency, list):
                consistency = sum(consistency) / len(consistency)
            items.append({
                "query": "Summarize the key information in the following article.",
                "response": summary,
                "context": [article[:1500]],
                "human_score": float(consistency),
                "source": "summeval",
            })
            if n is not None and len(items) >= n:
                break
        if n is not None and len(items) >= n:
            break

    print(f"Loaded {len(items)} examples from SummEval")
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_all(items: list[dict]) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot import Auditor

    auditor = Auditor()
    records = []
    t0 = time.perf_counter()

    for i, item in enumerate(items):
        ctx = item["context"]
        if isinstance(ctx, str):
            ctx = [ctx]
        result = auditor.score(
            query=item["query"],
            response=item["response"],
            context=ctx,
        )
        records.append({
            "human_score": item["human_score"],
            "source": item.get("source", "unknown"),
            "iqs": result.iqs,
            "groundedness": result.groundedness if result.groundedness is not None else 0.0,
            "completeness": result.completeness,
            "relevance": result.relevance,
            "consistency": result.consistency,
            "confidence": result.confidence,
            "flags": result.flags,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(items)}  {rate:.1f} calls/s  "
                  f"ETA {(len(items)-i-1)/rate:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(items):.2f}s/call)")
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _pearson(x: list[float], y: list[float]) -> tuple[float, float | None]:
    try:
        from scipy.stats import pearsonr
        r, pval = pearsonr(x, y)
        return float(r), float(pval)
    except ImportError:
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        dx = sum((x[i] - mx) ** 2 for i in range(n)) ** 0.5
        dy = sum((y[i] - my) ** 2 for i in range(n)) ** 0.5
        r = num / (dx * dy) if dx * dy > 0 else 0.0
        return float(r), None


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


def _compute_stats(records: list[dict]) -> dict:
    human = [r["human_score"] for r in records]
    iqs = [r["iqs"] for r in records]
    groundedness = [r["groundedness"] for r in records]

    # Normalise human 1–5 → 0–1 for MAE
    h_min = min(human)
    h_max = max(human)
    human_norm = [(h - h_min) / (h_max - h_min) if h_max > h_min else 0.5
                  for h in human]
    mae = sum(abs(h - q) for h, q in zip(human_norm, iqs)) / len(iqs)

    pearson_r, pearson_p = _pearson(human, iqs)
    spearman_r, spearman_p = _spearman(human, iqs)
    ground_r, _ = _pearson(human, groundedness)

    return {
        "benchmark": "human_correlation",
        "dataset": records[0].get("source", "unknown") if records else "unknown",
        "total_examples": len(records),
        "correlations": {
            "iqs_vs_human_pearson": {
                "r": round(pearson_r, 4),
                "p_value": round(pearson_p, 6) if pearson_p is not None else None,
                "target": f"> {TARGET_PEARSON}",
                "passed": pearson_r > TARGET_PEARSON,
            },
            "iqs_vs_human_spearman": {
                "r": round(spearman_r, 4),
                "p_value": round(spearman_p, 6) if spearman_p is not None else None,
            },
            "groundedness_vs_human_pearson": round(ground_r, 4),
        },
        "mean_absolute_error": round(mae, 4),
        "passed": pearson_r > TARGET_PEARSON,
    }


def _print_results(stats: dict) -> None:
    c = stats["correlations"]
    print("\n" + "─" * 58)
    print(f"  {'Metric':<42}  {'Value':>7}")
    print("─" * 58)
    print(f"  {'IQS vs human - Pearson r':42s}  "
          f"{c['iqs_vs_human_pearson']['r']:>+7.4f}")
    print(f"  {'IQS vs human - Spearman ρ':42s}  "
          f"{c['iqs_vs_human_spearman']['r']:>+7.4f}")
    print(f"  {'Groundedness vs human - Pearson r':42s}  "
          f"{c['groundedness_vs_human_pearson']:>+7.4f}")
    print(f"  {'MAE (normalised human 1–5 → 0–1)':42s}  "
          f"{stats['mean_absolute_error']:>7.4f}")
    print("─" * 58)
    r = c["iqs_vs_human_pearson"]["r"]
    passed = c["iqs_vs_human_pearson"]["passed"]
    pval = c["iqs_vs_human_pearson"].get("p_value")
    print(f"\n  Primary: IQS ↔ human  Pearson r = {r:+.4f}")
    print(f"  Target:  r > {TARGET_PEARSON}  →  {'PASS ✓' if passed else 'FAIL ✗'}")
    if pval is not None:
        print(f"  p-value: {pval:.4f}")
    print(f"  N = {stats['total_examples']}  dataset = {stats['dataset']}\n")


def _plot(records: list[dict], r: float, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot.")
        return

    human = [rec["human_score"] for rec in records]
    iqs = [rec["iqs"] for rec in records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(human, iqs, alpha=0.55, s=25, c="#2196F3", edgecolors="white",
               linewidths=0.5)
    ax.set_xlabel("Human Quality Score (1–5)")
    ax.set_ylabel("scroot IQS")
    ax.set_title(f"IQS vs Human Score  (Pearson r = {r:+.3f})")

    try:
        import numpy as np
        z = np.polyfit(human, iqs, 1)
        p = np.poly1d(z)
        x_line = sorted(set(human))
        ax.plot(x_line, [p(x) for x in x_line], "r--", linewidth=1.5,
                label=f"trend  r={r:+.3f}")
        ax.legend(fontsize=9)
    except Exception:
        pass

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    use_summeval: bool = False,
    n: int | None = None,
    from_cache: bool = False,
    no_plot: bool = False,
) -> dict:
    if from_cache and OUTPUT_PATH.exists():
        with OUTPUT_PATH.open() as f:
            stats = json.load(f)
        print(f"Loaded cached results from {OUTPUT_PATH}")
        _print_results(stats)
        return stats

    if not use_summeval and COMMITTED_FILE.exists():
        items = _load_committed(COMMITTED_FILE)
    else:
        if not use_summeval:
            print(f"Committed file not found at {COMMITTED_FILE} - "
                  f"falling back to SummEval.")
        items = _load_summeval(n)

    if not items:
        print("ERROR: no items loaded.", file=sys.stderr)
        sys.exit(1)

    records = _score_all(items)

    stats = _compute_stats(records)
    _print_results(stats)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Results → {OUTPUT_PATH}")

    if not no_plot:
        _plot(records, stats["correlations"]["iqs_vs_human_pearson"]["r"], PLOT_PATH)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--use-summeval", action="store_true",
                        help="Force SummEval even if committed file exists")
    parser.add_argument("--n", type=int, default=None,
                        help="Max SummEval examples (ignored for committed file)")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    result = run(
        use_summeval=args.use_summeval,
        n=args.n,
        from_cache=args.from_cache,
        no_plot=args.no_plot,
    )
    sys.exit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
