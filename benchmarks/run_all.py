"""Run all benchmarks and generate a summary report.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --skip-slow        # claim + flag only (CI-friendly)
    python -m benchmarks.run_all --only speed        # single benchmark
    python -m benchmarks.run_all --only claim_accuracy,flag_accuracy

Exit code: 0 if all run benchmarks passed, 1 if any failed.

Dataset prerequisites (run once before the slow benchmarks):
    python benchmarks/datasets/generate_nq.py
    python benchmarks/datasets/generate_perturbations.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.md"

# Registry: (module_path, is_slow)
# is_slow=True means the benchmark loads models or downloads data
_BENCHMARKS: dict[str, tuple[str, bool]] = {
    # Fast - no models needed
    "flag_accuracy":             ("benchmarks.bench_flag_accuracy",             False),
    "claim_accuracy":            ("benchmarks.bench_claim_accuracy",            False),
    "confidence_accuracy":       ("benchmarks.bench_confidence_accuracy",       False),
    "feedback_loop":             ("benchmarks.bench_feedback_loop",             False),
    # Fast - embedding model only
    "completeness_accuracy":     ("benchmarks.bench_completeness_accuracy",     True),
    "paraphrase_groundedness":   ("benchmarks.bench_paraphrase_groundedness",   True),
    # Slow - NLI + embedding + large datasets
    "determinism":               ("benchmarks.bench_determinism",               True),
    "speed":                     ("benchmarks.bench_speed",                     True),
    "correlation":               ("benchmarks.bench_correlation",               True),
    "human_correlation":         ("benchmarks.bench_vs_human",                  True),
    "competitors":               ("benchmarks.bench_vs_competitors",            True),
}


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _generate_summary(all_results: dict[str, dict | None]) -> None:
    lines = [
        "# scroot Benchmark Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Summary table",
        "",
        "| Benchmark | Status | Key metric |",
        "|---|---|---|",
    ]

    for name, result in all_results.items():
        if result is None:
            lines.append(f"| {name} | SKIPPED | - |")
            continue
        if "error" in result:
            lines.append(f"| {name} | ERROR | {result['error'][:60]} |")
            continue

        passed = result.get("passed")
        status = "PASS ✓" if passed else ("FAIL ✗" if passed is False else "—")

        if name == "correlation":
            corr = result.get("correlations", {}).get("iqs_vs_perturbation", {})
            key = f"ρ = {corr.get('spearman_r', '?')}"
        elif name == "human_correlation":
            corr = result.get("correlations", {}).get("iqs_vs_human_pearson", {})
            key = f"r = {corr.get('r', '?')}"
        elif name == "speed":
            m = result.get("measurements", {})
            key = (f"import {m.get('import_time_ms','?')}ms  "
                   f"short+1ctx {m.get('cpu_short_1_context',{}).get('mean_ms','?')}ms")
        elif name == "determinism":
            dev = result.get("deviations_found", "?")
            key = f"{dev} deviations"
        elif name == "competitors":
            scroot = next(
                (c for c in result.get("comparisons", {}).values()
                 if isinstance(c, dict) and c.get("framework") == "scroot"), {}
            )
            key = f"scroot ρ = {scroot.get('spearman_r', '?')}"
        elif name == "claim_accuracy":
            key = (f"P={result.get('precision','?')}  "
                   f"R={result.get('recall','?')}  "
                   f"F1={result.get('f1','?')}")
        elif name == "flag_accuracy":
            key = (f"P={result.get('precision','?')}  "
                   f"R={result.get('recall','?')}  "
                   f"case-acc={result.get('case_accuracy','?')}")
        else:
            key = "—"

        lines.append(f"| {name} | {status} | {key} |")

    lines += [
        "",
        "## Detail",
        "",
    ]

    for name, result in all_results.items():
        lines.append(f"### {name}")
        if result is None:
            lines.append("SKIPPED\n")
            continue
        if "error" in result:
            lines.append(f"ERROR: {result['error']}\n")
            continue

        # Dump a concise subset of the result as JSON
        condensed = {k: v for k, v in result.items()
                     if k not in ("details", "deviations", "scored_items")}
        lines.append("```json")
        lines.append(json.dumps(condensed, indent=2))
        lines.append("```")
        lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary → {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-slow", action="store_true",
        help="Skip model-loading benchmarks; run only flag_accuracy and claim_accuracy",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated list of benchmark names to run (e.g. 'speed,determinism')",
    )
    args = parser.parse_args()

    only_set: set[str] | None = None
    if args.only:
        only_set = {n.strip() for n in args.only.split(",")}
        unknown = only_set - set(_BENCHMARKS)
        if unknown:
            print(f"ERROR: unknown benchmarks: {unknown}", file=sys.stderr)
            print(f"Available: {list(_BENCHMARKS)}", file=sys.stderr)
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Add project src to path so benchmarks can import scroot
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    all_results: dict[str, dict | None] = {}
    any_failed = False

    for name, (module_path, is_slow) in _BENCHMARKS.items():
        if only_set is not None and name not in only_set:
            continue
        if args.skip_slow and is_slow:
            print(f"\n{'='*60}")
            print(f"SKIPPING {name}  (--skip-slow)")
            all_results[name] = None
            continue

        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        try:
            module = importlib.import_module(module_path)
            result = module.run()
            elapsed = time.perf_counter() - t0
            all_results[name] = result

            passed = result.get("passed") if result else None
            status = "PASS ✓" if passed else ("FAIL ✗" if passed is False else "done")
            print(f"\n[{name}] {status}  ({elapsed:.1f}s)")

            if passed is False:
                any_failed = True

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"\nERROR in {name}: {exc}  ({elapsed:.1f}s)")
            all_results[name] = {"error": str(exc), "passed": False}
            any_failed = True

    _generate_summary(all_results)

    print(f"\n{'='*60}")
    passed_count = sum(1 for r in all_results.values()
                       if r and r.get("passed") is True)
    failed_count = sum(1 for r in all_results.values()
                       if r and r.get("passed") is False)
    skipped_count = sum(1 for r in all_results.values() if r is None)
    print(f"Results: {passed_count} passed  {failed_count} failed  "
          f"{skipped_count} skipped")
    print(f"{'='*60}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
