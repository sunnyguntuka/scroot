"""
Benchmark: Determinism verification.

Scores the same 100 examples 10 times each and asserts that every run
produces bit-for-bit identical IQS (and all sub-metric) scores.

Why this matters: sentence-transformers + numpy should be fully deterministic
on CPU for the same input. Any deviation indicates a non-deterministic path
(e.g. random dropout left on, ThreadPool race condition).

Pass criterion: 100% of (example, metric) pairs are identical across all runs.

Usage:
    python benchmarks/bench_determinism.py
    python benchmarks/bench_determinism.py --n 20 --runs 5
    python benchmarks/bench_determinism.py --dataset path/to/nq_500.jsonl

Output:
    "100% deterministic" or a table of deviating (example, metric) pairs
    benchmarks/results/determinism_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Prevent benchmarks/ (which contains a datasets/ package) from shadowing the
# HuggingFace `datasets` package that sentence-transformers imports. Other bench
# scripts do this; bench_determinism.py historically did not, so running it with
# the project root or benchmarks/ on sys.path broke `from datasets import Dataset`.
_bench_dir = str(Path(__file__).parent)
while _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "nq_500_perturbed.jsonl"

N_EXAMPLES = 100
N_RUNS = 10
METRICS = ["iqs", "groundedness", "completeness", "relevance",
           "consistency", "confidence"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_examples(path: Path, n: int) -> list[dict]:
    if not path.exists():
        print(f"ERROR: dataset not found at {path}\n"
              f"Run:  python benchmarks/datasets/prepare_nq.py", file=sys.stderr)
        sys.exit(1)
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Two dataset schemas are supported:
            #  (a) legacy nested: {"question", "context", "perturbations":{"A0":{"response"}}}
            #  (b) current flat one-row-per-(id,level):
            #      {"id","query","context","response","perturbation_level"}
            # For (b) we keep only the clean A0 rows (perturbation_level == 0)
            # to mirror the legacy "highest-quality response" choice.
            if "perturbations" in ex:
                examples.append({
                    "id": ex["id"],
                    "query": ex.get("question", ex.get("query", "")),
                    "response": ex["perturbations"]["A0"]["response"],
                    "context": [ex["context"]],
                })
            else:
                if ex.get("perturbation_level", 0) != 0:
                    continue
                examples.append({
                    "id": ex["id"],
                    "query": ex.get("query", ex.get("question", "")),
                    "response": ex["response"],
                    "context": [ex["context"]],
                })
            if len(examples) >= n:
                break
    return examples


def _score_batch(auditor, examples: list[dict]) -> list[dict]:
    """Score all examples and return flat list of metric dicts."""
    records = []
    for ex in examples:
        result = auditor.score(
            query=ex["query"],
            response=ex["response"],
            context=ex["context"],
        )
        records.append({
            "id": ex["id"],
            "iqs": result.iqs,
            "groundedness": result.groundedness,
            "completeness": result.completeness,
            "relevance": result.relevance,
            "consistency": result.consistency,
            "confidence": result.confidence,
        })
    return records


def _compare_runs(all_runs: list[list[dict]]) -> list[dict]:
    """Return list of deviations across runs. Empty list = fully deterministic."""
    deviations = []
    baseline = all_runs[0]
    for run_idx, run in enumerate(all_runs[1:], start=2):
        for ex_idx, (base_rec, run_rec) in enumerate(zip(baseline, run)):
            for metric in METRICS:
                base_val = base_rec.get(metric)
                run_val = run_rec.get(metric)
                # Both None is fine
                if base_val is None and run_val is None:
                    continue
                # One None, other not: deviation
                if (base_val is None) != (run_val is None):
                    deviations.append({
                        "example_id": base_rec["id"],
                        "example_index": ex_idx,
                        "run": run_idx,
                        "metric": metric,
                        "baseline": base_val,
                        "deviation": run_val,
                    })
                    continue
                # Both floats: must be exactly equal
                if base_val != run_val:
                    deviations.append({
                        "example_id": base_rec["id"],
                        "example_index": ex_idx,
                        "run": run_idx,
                        "metric": metric,
                        "baseline": base_val,
                        "deviation": run_val,
                        "abs_diff": abs(base_val - run_val),
                    })
    return deviations


def _print_results(deviations: list[dict], n_examples: int, n_runs: int) -> None:
    total_checks = n_examples * len(METRICS) * (n_runs - 1)
    n_deviations = len(deviations)
    det_rate = (total_checks - n_deviations) / total_checks * 100

    print("\n" + "─" * 60)
    print(f"  Examples:     {n_examples}")
    print(f"  Runs:         {n_runs}")
    print(f"  Metrics:      {len(METRICS)}")
    print(f"  Total checks: {total_checks:,}")
    print(f"  Deviations:   {n_deviations}")
    print(f"  Determinism:  {det_rate:.2f}%")
    print("─" * 60)

    if not deviations:
        print("\n  ✓  100% deterministic - every run produced identical scores.\n")
    else:
        print(f"\n  ✗  {n_deviations} deviation(s) detected:\n")
        for d in deviations[:20]:   # show at most 20
            print(f"    example {d['example_index']}  run {d['run']}  "
                  f"{d['metric']}: "
                  f"{d['baseline']} → {d['deviation']}"
                  + (f"  Δ={d['abs_diff']:.2e}" if "abs_diff" in d else ""))
        if len(deviations) > 20:
            print(f"    ... and {len(deviations) - 20} more (see JSON output)")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n: int = N_EXAMPLES, runs: int = N_RUNS) -> dict:
    """Entry point for benchmarks.run_all. Uses synthetic examples (no dataset needed)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot import Auditor

    examples = [
        {
            "id": f"synthetic_{i:03d}",
            "query": f"Question number {i} about policies?",
            "response": f"The answer to question {i} involves specific policy details.",
            "context": [f"Policy {i} states specific rules about procedures."],
        }
        for i in range(n)
    ]
    print(f"Running determinism check: {n} examples x {runs} passes...")

    auditor = Auditor()
    auditor.score(query="test", response="test", context=["test"])  # warm up

    all_runs: list[list[dict]] = []
    for run_idx in range(runs):
        t0 = time.perf_counter()
        records = _score_batch(auditor, examples)
        elapsed = time.perf_counter() - t0
        all_runs.append(records)
        print(f"  Run {run_idx + 1}/{runs} complete  ({elapsed:.1f}s)")

    deviations = _compare_runs(all_runs)
    _print_results(deviations, n, runs)

    total_checks = n * len(METRICS) * (runs - 1)
    n_dev = len(deviations)
    passed = n_dev == 0

    out = RESULTS_DIR / "determinism_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with out.open("w") as f:
        json.dump({
            "n_examples": n,
            "n_runs": runs,
            "n_metrics": len(METRICS),
            "total_checks": total_checks,
            "n_deviations": n_dev,
            "determinism_rate_pct": round((total_checks - n_dev) / total_checks * 100, 4),
            "passed": passed,
            "deviations": deviations[:20],
        }, f, indent=2)
    return {
        "benchmark": "determinism",
        "n_examples": n,
        "n_runs": runs,
        "total_checks": total_checks,
        "deviations_found": n_dev,
        "is_deterministic": passed,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=N_EXAMPLES,
                        help=f"Number of examples to score (default: {N_EXAMPLES})")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Number of repeated scoring runs (default: {N_RUNS})")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot import Auditor

    examples = _load_examples(Path(args.dataset), args.n)
    print(f"Loaded {len(examples)} examples from {args.dataset}")
    print(f"Running {args.runs} scoring passes...\n")

    auditor = Auditor()

    # Warm up
    auditor.score(query="test", response="test", context=["test"])

    all_runs: list[list[dict]] = []
    for run_idx in range(args.runs):
        t0 = time.perf_counter()
        records = _score_batch(auditor, examples)
        elapsed = time.perf_counter() - t0
        all_runs.append(records)
        print(f"  Run {run_idx + 1}/{args.runs} complete  ({elapsed:.1f}s)")

    deviations = _compare_runs(all_runs)
    _print_results(deviations, len(examples), args.runs)

    total_checks = len(examples) * len(METRICS) * (args.runs - 1)
    n_dev = len(deviations)
    passed = n_dev == 0

    out = RESULTS_DIR / "determinism_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "n_examples": len(examples),
            "n_runs": args.runs,
            "n_metrics": len(METRICS),
            "total_checks": total_checks,
            "n_deviations": n_dev,
            "determinism_rate_pct": round((total_checks - n_dev) / total_checks * 100, 4),
            "passed": passed,
            "deviations": deviations[:100],   # cap JSON size
        }, f, indent=2)
    print(f"Results → {out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
