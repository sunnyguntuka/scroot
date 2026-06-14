"""
Benchmark: Completeness metric accuracy on multi-question queries.

Tests whether score_completeness() correctly measures how well a response
covers the different aspects of a multi-part query.

Target: Spearman ρ > 0.80 between completeness scores and expected coverage.

No model loading required for collection - embedding model needed for scoring.

Usage:
    python benchmarks/bench_completeness_accuracy.py

Output:
    benchmarks/results/completeness_accuracy.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "completeness_accuracy.json"

# ---------------------------------------------------------------------------
# Labeled dataset
# label: fraction of query aspects the response addresses (0.0 – 1.0)
# ---------------------------------------------------------------------------

COMPLETENESS_CASES = [
    # --- Full coverage (label = 1.0) ---
    {
        "query": "What is the refund policy and how long does shipping take?",
        "response": "We offer a 30-day full refund for all purchases. Standard shipping takes 5-7 business days.",
        "label": 1.0,
        "note": "both refund policy and shipping time answered",
    },
    {
        "query": "What are the pricing tiers and which features are included in the Pro plan?",
        "response": "We have three tiers: Basic ($9), Pro ($29), and Enterprise ($99). The Pro plan includes unlimited API calls, priority support, and advanced analytics.",
        "label": 1.0,
        "note": "pricing and Pro features both covered",
    },
    {
        "query": "How do I reset my password and what should I do if I don't receive the email?",
        "response": "Click 'Forgot Password' on the login page and enter your email. If you don't receive the email within 5 minutes, check your spam folder or contact support.",
        "label": 1.0,
        "note": "both reset steps and email fallback covered",
    },

    # --- Partial coverage (label ~0.5) ---
    {
        "query": "What is the refund policy and how long does shipping take?",
        "response": "We offer a 30-day full refund for all purchases, no questions asked.",
        "label": 0.5,
        "note": "only refund policy answered, shipping not mentioned",
    },
    {
        "query": "What are the system requirements and how do I install the software?",
        "response": "The software requires Windows 10 or macOS 11, 8GB RAM, and 2GB disk space.",
        "label": 0.5,
        "note": "only system requirements, no installation steps",
    },
    {
        "query": "What payment methods are accepted and is there a free trial available?",
        "response": "We accept all major credit cards including Visa, Mastercard, and American Express.",
        "label": 0.5,
        "note": "payment methods answered, free trial not mentioned",
    },
    {
        "query": "How do I contact support and what are the support hours?",
        "response": "You can reach our support team at support@example.com.",
        "label": 0.5,
        "note": "contact info provided, hours not mentioned",
    },

    # --- No coverage (label = 0.0 or very low) ---
    {
        "query": "What is the refund policy and how long does shipping take?",
        "response": "Our products are manufactured using the highest quality materials.",
        "label": 0.0,
        "note": "neither refund nor shipping mentioned",
    },
    {
        "query": "How do I upgrade my account and what are the benefits?",
        "response": "We have been serving customers for over 10 years with excellence.",
        "label": 0.0,
        "note": "completely off-topic",
    },
    {
        "query": "What programming languages are supported and is there an SDK available?",
        "response": "Our pricing starts at $9 per month for the basic plan.",
        "label": 0.0,
        "note": "answers about pricing, not languages or SDK",
    },
]

TARGET_RHO = 0.80


def _spearman(x: list[float], y: list[float]) -> float:
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except ImportError:
        def _rank(vals: list[float]) -> list[float]:
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
        return num / (dx * dy) if dx * dy > 0 else 0.0


def run() -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot.metrics.completeness import score_completeness

    scored_labels = []
    scored_scores = []
    details = []

    for case in COMPLETENESS_CASES:
        score, det = score_completeness(case["query"], case["response"])
        scored_scores.append(score)
        scored_labels.append(case["label"])
        details.append({
            "query": case["query"],
            "response": case["response"][:80] + "...",
            "expected_label": case["label"],
            "completeness_score": round(score, 4),
            "delta": round(abs(score - case["label"]), 4),
            "n_aspects": det.get("total_segments", "?"),
            "covered": det.get("covered_segments", "?"),
            "note": case.get("note", ""),
        })

    rho = _spearman(scored_labels, scored_scores)
    mae = sum(abs(s - l) for s, l in zip(scored_scores, scored_labels)) / len(scored_scores)
    passed = rho >= TARGET_RHO

    results = {
        "benchmark": "completeness_accuracy",
        "n_cases": len(COMPLETENESS_CASES),
        "spearman_rho": round(rho, 4),
        "mae": round(mae, 4),
        "target_rho": TARGET_RHO,
        "passed": passed,
        "details": details,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Completeness Metric Accuracy ===")
    print(f"  N cases:      {len(COMPLETENESS_CASES)}")
    print(f"  Spearman ρ:   {rho:+.4f}  (target: >= {TARGET_RHO})")
    print(f"  MAE:          {mae:.4f}")
    print(f"  Passed:       {'YES ✓' if passed else 'NO ✗'}")
    print(f"  Results →     {OUTPUT_PATH}")

    if not passed:
        print("\n  Worst cases:")
        worst = sorted(details, key=lambda d: -d["delta"])[:3]
        for d in worst:
            print(f"    delta={d['delta']:.3f}  got={d['completeness_score']:.3f}"
                  f"  expected={d['expected_label']}  ({d['note']})")

    return results


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["passed"] else 1)
