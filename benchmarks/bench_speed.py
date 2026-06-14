"""
Benchmark: Latency and memory profiling.

Measures wall-clock latency and peak RSS memory across all meaningful
scoring scenarios, matching the expected README performance table.

Scenarios timed:
  1. import scroot              (subprocess, model-free)
  2. First score() call           (cold: includes model weight loading)
  3. Short response, no context   (embeddings + regex only)
  4. Short response, 1 context    (+ NLI for groundedness)
  5. Medium response, 1 context
  6. Long response, 1 context
  7. Short response, 10 contexts  (batched NLI scales linearly)
  8. Short response, 50 contexts
  9. score_batch(100 items)
  10. Consistency isolation: 10, 25, 50 sentences

Methodology:
  - Each scenario: WARMUP warm-up calls (excluded) + REPS timed calls.
  - Memory: tracemalloc heap delta + psutil RSS delta (if available).
  - GPU benchmarked automatically if torch.cuda.is_available().

Output:
    Markdown table printed to stdout
    benchmarks/results/speed.json

Usage:
    python benchmarks/bench_speed.py
    python benchmarks/bench_speed.py --reps 3
    python benchmarks/bench_speed.py --no-gpu
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "speed.json"

WARMUP = 1
REPS = 5   # default; --reps overrides

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SHORT_QUERY = "What is our refund policy?"
_SHORT_RESPONSE = "We offer a 30-day full refund at no extra cost."
_SHORT_CONTEXT_1 = [
    "All customers are eligible for a 30-day full refund at no extra cost."
]
_SHORT_CONTEXT_10 = [
    f"Policy clause {i}: customers may return items under certain conditions."
    for i in range(10)
] + _SHORT_CONTEXT_1

_SHORT_CONTEXT_50 = [
    f"Policy document section {i}: further details on eligibility criteria."
    for i in range(49)
] + _SHORT_CONTEXT_1

_MEDIUM_RESPONSE = (
    "We offer a 30-day full refund at no extra cost. "
    "Simply contact our support team to initiate the return. "
    "The refund will be processed within 5-7 business days. "
    "Original shipping costs are non-refundable. "
    "Items must be returned in their original packaging."
)

_LONG_RESPONSE = (
    "Our comprehensive returns and refund policy provides customers with "
    "a hassle-free experience. Standard customers may return eligible items "
    "within 30 calendar days of the original purchase date. "
    "The item must be in its original condition with all tags and packaging. "
    "Items that have been used or damaged by the customer do not qualify "
    "for a full refund. "
    "To initiate a return, contact our support team through the online portal "
    "and provide your order number and reason for return. "
    "Once approved, a prepaid shipping label will be issued for defective "
    "or incorrect items. "
    "Refunds are processed within 5 business days of receiving the item. "
    "The refund is issued to the original payment method. "
    "Processing times may vary depending on the customer's bank. "
    "Premium members enjoy a 60-day return window and expedited processing "
    "within 2 business days. "
    "Certain categories are excluded, including perishable goods, digital "
    "downloads, and personalised items. "
    "International orders may take up to 10 business days for refund processing. "
    "Gift cards and vouchers are non-refundable under any circumstances. "
    "Products from third-party sellers are subject to their individual policies. "
    "For further questions, please contact our customer support team directly."
)

_CONSISTENCY_SENTENCES = {
    10:  " ".join(f"Fact {i} about our policy is definitively correct." for i in range(10)),
    25:  " ".join(f"Fact {i} about our policy is definitively correct." for i in range(25)),
    50:  " ".join(f"Fact {i} about our policy is definitively correct." for i in range(50)),
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, warmup: int = WARMUP, reps: int = REPS) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    return {
        "mean_ms": round(mean * 1000, 1),
        "std_ms": round(variance ** 0.5 * 1000, 1),
        "min_ms": round(min(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
        "runs": reps,
    }


def _measure_import_ms() -> float:
    import subprocess
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'src'); import scroot"],
        capture_output=True,
        cwd=str(Path(__file__).parent.parent),
    )
    return (time.perf_counter() - t0) * 1000


def _heap_delta_mb(fn) -> float:
    import tracemalloc
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def _rss_mb() -> float | None:
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_speed_benchmark(reps: int = REPS, use_gpu: bool = True) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot import Auditor
    from scroot.metrics.consistency import score_consistency

    results: dict = {"benchmark": "speed", "measurements": {}}
    m = results["measurements"]

    # ── 1. Import time ──────────────────────────────────────────────────────
    print("Measuring import time (subprocess)...")
    import_ms = _measure_import_ms()
    m["import_time_ms"] = round(import_ms, 1)
    print(f"  import scroot: {import_ms:.0f}ms")

    devices = ["cpu"]
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                devices.append("cuda")
                print("GPU detected - will benchmark both CPU and CUDA.")
            else:
                print("GPU not available (torch.cuda.is_available()=False). CPU only.")
        except ImportError:
            print("torch not installed - CPU only.")

    for device in devices:
        print(f"\n{'─'*52}")
        print(f"Device: {device.upper()}")
        print(f"{'─'*52}")
        prefix = f"{device}_"

        auditor = Auditor(device=device)

        # ── 2. First (cold) score() call ────────────────────────────────────
        print("  Cold call (model loading)...")
        rss_before = _rss_mb()
        t_cold = time.perf_counter()
        auditor.score(query=_SHORT_QUERY, response=_SHORT_RESPONSE,
                      context=_SHORT_CONTEXT_1)
        cold_ms = (time.perf_counter() - t_cold) * 1000
        rss_after_load = _rss_mb()
        m[f"{prefix}first_call_ms"] = round(cold_ms, 1)
        if rss_before is not None and rss_after_load is not None:
            m[f"{prefix}peak_rss_after_model_load_mb"] = round(rss_after_load, 0)
        print(f"  First call: {cold_ms:.0f}ms")

        # ── 3. Short, no context ─────────────────────────────────────────────
        print(f"  Short / no context  ({reps} reps)...")
        t = _time_fn(lambda: auditor.score(
            query=_SHORT_QUERY, response=_SHORT_RESPONSE), reps=reps)
        heap = _heap_delta_mb(lambda: auditor.score(
            query=_SHORT_QUERY, response=_SHORT_RESPONSE))
        m[f"{prefix}short_no_context"] = {**t, "peak_heap_mb": round(heap, 1)}
        print(f"    {t['mean_ms']}ms ± {t['std_ms']}ms  heap {heap:.1f}MB")

        # ── 4. Short, 1 context ──────────────────────────────────────────────
        print(f"  Short / 1 context   ({reps} reps)...")
        t = _time_fn(lambda: auditor.score(
            query=_SHORT_QUERY, response=_SHORT_RESPONSE,
            context=_SHORT_CONTEXT_1), reps=reps)
        heap = _heap_delta_mb(lambda: auditor.score(
            query=_SHORT_QUERY, response=_SHORT_RESPONSE,
            context=_SHORT_CONTEXT_1))
        m[f"{prefix}short_1_context"] = {**t, "peak_heap_mb": round(heap, 1)}
        print(f"    {t['mean_ms']}ms ± {t['std_ms']}ms  heap {heap:.1f}MB")

        # ── 5. Medium, 1 context ─────────────────────────────────────────────
        print(f"  Medium / 1 context  ({reps} reps)...")
        t = _time_fn(lambda: auditor.score(
            query=_SHORT_QUERY, response=_MEDIUM_RESPONSE,
            context=_SHORT_CONTEXT_1), reps=reps)
        heap = _heap_delta_mb(lambda: auditor.score(
            query=_SHORT_QUERY, response=_MEDIUM_RESPONSE,
            context=_SHORT_CONTEXT_1))
        m[f"{prefix}medium_1_context"] = {**t, "peak_heap_mb": round(heap, 1)}
        print(f"    {t['mean_ms']}ms ± {t['std_ms']}ms  heap {heap:.1f}MB")

        # ── 6. Long, 1 context ───────────────────────────────────────────────
        print(f"  Long / 1 context    ({reps} reps)...")
        t = _time_fn(lambda: auditor.score(
            query=_SHORT_QUERY, response=_LONG_RESPONSE,
            context=_SHORT_CONTEXT_1), reps=reps)
        heap = _heap_delta_mb(lambda: auditor.score(
            query=_SHORT_QUERY, response=_LONG_RESPONSE,
            context=_SHORT_CONTEXT_1))
        m[f"{prefix}long_1_context"] = {**t, "peak_heap_mb": round(heap, 1)}
        print(f"    {t['mean_ms']}ms ± {t['std_ms']}ms  heap {heap:.1f}MB")

        # ── 7. Short, 10 contexts ────────────────────────────────────────────
        print(f"  Short / 10 contexts ({reps} reps)...")
        t = _time_fn(lambda: auditor.score(
            query=_SHORT_QUERY, response=_SHORT_RESPONSE,
            context=_SHORT_CONTEXT_10), reps=reps)
        m[f"{prefix}short_10_context"] = t
        print(f"    {t['mean_ms']}ms ± {t['std_ms']}ms")

        # ── 8. Short, 50 contexts ────────────────────────────────────────────
        print(f"  Short / 50 contexts ({max(2,reps//2)} reps)...")
        t = _time_fn(lambda: auditor.score(
            query=_SHORT_QUERY, response=_SHORT_RESPONSE,
            context=_SHORT_CONTEXT_50), reps=max(2, reps // 2))
        m[f"{prefix}short_50_context"] = t
        print(f"    {t['mean_ms']}ms ± {t['std_ms']}ms")

        # ── 9. Batch 100 items ───────────────────────────────────────────────
        print("  score_batch(100) ...")
        batch_items = [
            {"query": _SHORT_QUERY, "response": _SHORT_RESPONSE,
             "context": _SHORT_CONTEXT_1}
            for _ in range(100)
        ]
        t0 = time.perf_counter()
        auditor.score_batch(batch_items)
        batch_ms = (time.perf_counter() - t0) * 1000
        m[f"{prefix}batch_100_ms"] = round(batch_ms, 1)
        m[f"{prefix}batch_100_per_item_ms"] = round(batch_ms / 100, 1)
        print(f"    total {batch_ms:.0f}ms  ({batch_ms/100:.1f}ms/item)")

        # ── 10. Consistency isolation (10 / 25 / 50 sentences) ──────────────
        for n_sent, text in _CONSISTENCY_SENTENCES.items():
            print(f"  Consistency {n_sent} sentences ({max(2,reps//2)} reps)...")
            def _score_consistency(t=text):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score_consistency(t, nli_model="cross-encoder/nli-deberta-v3-base",
                                      device=device)
            t = _time_fn(_score_consistency, warmup=1, reps=max(2, reps // 2))
            m[f"{prefix}consistency_{n_sent}_sentences"] = t
            print(f"    {t['mean_ms']}ms ± {t['std_ms']}ms")

        rss_final = _rss_mb()
        if rss_final is not None:
            m[f"{prefix}peak_rss_final_mb"] = round(rss_final, 0)

    return results


# ---------------------------------------------------------------------------
# Markdown table output
# ---------------------------------------------------------------------------

def _print_markdown(results: dict) -> None:
    m = results["measurements"]
    cpu = m.get

    print("\n\n## Speed Benchmark Results\n")
    print(f"Import: **{m.get('import_time_ms', '?'):.0f}ms**\n")

    rows = [
        ("Operation", "CPU", "GPU"),
        ("---", "---", "---"),
        ("`import scroot`",
         f"{m.get('import_time_ms','?')}ms", "—"),
        ("First `score()` (model load)",
         f"~{m.get('cpu_first_call_ms','?')}ms",
         f"~{m.get('cuda_first_call_ms','—')}ms" if "cuda_first_call_ms" in m else "—"),
        ("Short, no context",
         f"{m.get('cpu_short_no_context',{}).get('mean_ms','?')}ms",
         f"{m.get('cuda_short_no_context',{}).get('mean_ms','—')}ms"
         if "cuda_short_no_context" in m else "—"),
        ("Short, 1 context",
         f"{m.get('cpu_short_1_context',{}).get('mean_ms','?')}ms",
         f"{m.get('cuda_short_1_context',{}).get('mean_ms','—')}ms"
         if "cuda_short_1_context" in m else "—"),
        ("Medium, 1 context",
         f"{m.get('cpu_medium_1_context',{}).get('mean_ms','?')}ms",
         f"{m.get('cuda_medium_1_context',{}).get('mean_ms','—')}ms"
         if "cuda_medium_1_context" in m else "—"),
        ("Long, 1 context",
         f"{m.get('cpu_long_1_context',{}).get('mean_ms','?')}ms",
         f"{m.get('cuda_long_1_context',{}).get('mean_ms','—')}ms"
         if "cuda_long_1_context" in m else "—"),
        ("Short, 10 contexts",
         f"{m.get('cpu_short_10_context',{}).get('mean_ms','?')}ms", "—"),
        ("Short, 50 contexts",
         f"{m.get('cpu_short_50_context',{}).get('mean_ms','?')}ms", "—"),
        ("score_batch(100)",
         f"{m.get('cpu_batch_100_ms','?')}ms total  "
         f"({m.get('cpu_batch_100_per_item_ms','?')}ms/item)", "—"),
        ("Consistency - 10 sentences",
         f"{m.get('cpu_consistency_10_sentences',{}).get('mean_ms','?')}ms", "—"),
        ("Consistency - 25 sentences",
         f"{m.get('cpu_consistency_25_sentences',{}).get('mean_ms','?')}ms", "—"),
        ("Consistency - 50 sentences",
         f"{m.get('cpu_consistency_50_sentences',{}).get('mean_ms','?')}ms", "—"),
        ("Peak RSS after model load",
         f"{m.get('cpu_peak_rss_after_model_load_mb','?')}MB", "—"),
    ]

    col_w = [max(len(str(r[i])) for r in rows) for i in range(3)]
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(col_w[i]) for i in range(3)) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    """Entry point for benchmarks.run_all."""
    results = run_speed_benchmark()
    _print_markdown(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    results = run_speed_benchmark(reps=args.reps, use_gpu=not args.no_gpu)
    _print_markdown(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
