"""
Benchmark: Flag detection precision and recall.

Tests detect_flags() against 9 hand-crafted score combinations covering:
  - Each individual flag in isolation
  - All flags simultaneously
  - No-context mode (groundedness=None suppresses groundedness flags)
  - Borderline cases just above and just below each threshold

Target: Precision > 0.90, Recall > 0.90

No model loading required - runs in < 0.1 second (pure logic).

Usage:
    python benchmarks/bench_flag_accuracy.py

Output:
    benchmarks/results/flag_accuracy.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "flag_accuracy.json"

# ---------------------------------------------------------------------------
# Test cases: (score dict) → expected flag set
# Thresholds from scroot/flags.py:
#   hallucination_risk:  groundedness < 0.5  AND  confidence > 0.7
#   off_topic:           relevance < 0.3
#   self_contradictory:  consistency < 0.7
#   incomplete:          completeness < 0.3
#   ungrounded:          groundedness < 0.3
# ---------------------------------------------------------------------------

FLAG_TEST_CASES = [
    # 1 - hallucination: high confidence, low groundedness
    {
        "label": "hallucination_risk only",
        "scores": {"groundedness": 0.2, "completeness": 0.8,
                   "relevance": 0.8, "consistency": 0.9, "confidence": 0.9},
        "expected_flags": ["hallucination_risk", "ungrounded"],
    },
    # 2 - clean response, nothing flagged
    {
        "label": "clean (no flags)",
        "scores": {"groundedness": 0.95, "completeness": 0.9,
                   "relevance": 0.9, "consistency": 0.95, "confidence": 0.7},
        "expected_flags": [],
    },
    # 3 - off-topic only
    {
        "label": "off_topic only",
        "scores": {"groundedness": 0.8, "completeness": 0.5,
                   "relevance": 0.1, "consistency": 0.9, "confidence": 0.5},
        "expected_flags": ["off_topic"],
    },
    # 4 - self-contradictory only
    {
        "label": "self_contradictory only",
        "scores": {"groundedness": 0.8, "completeness": 0.8,
                   "relevance": 0.8, "consistency": 0.3, "confidence": 0.5},
        "expected_flags": ["self_contradictory"],
    },
    # 5 - incomplete only
    {
        "label": "incomplete only",
        "scores": {"groundedness": 0.9, "completeness": 0.1,
                   "relevance": 0.8, "consistency": 0.9, "confidence": 0.5},
        "expected_flags": ["incomplete"],
    },
    # 6 - all flags simultaneously
    {
        "label": "all flags",
        "scores": {"groundedness": 0.1, "completeness": 0.1,
                   "relevance": 0.1, "consistency": 0.3, "confidence": 0.95},
        "expected_flags": ["hallucination_risk", "off_topic",
                           "self_contradictory", "incomplete", "ungrounded"],
    },
    # 7 - no context: groundedness=None suppresses hallucination_risk + ungrounded
    {
        "label": "no context (groundedness=None)",
        "scores": {"groundedness": None, "completeness": 0.8,
                   "relevance": 0.8, "consistency": 0.9, "confidence": 0.95},
        "expected_flags": [],
    },
    # 8 - borderline ABOVE all thresholds → no flags
    {
        "label": "borderline above thresholds",
        "scores": {"groundedness": 0.51, "completeness": 0.31,
                   "relevance": 0.31, "consistency": 0.71, "confidence": 0.5},
        "expected_flags": [],
    },
    # 9 - borderline BELOW all thresholds → multiple flags
    {
        "label": "borderline below thresholds",
        "scores": {"groundedness": 0.49, "completeness": 0.29,
                   "relevance": 0.29, "consistency": 0.69, "confidence": 0.71},
        "expected_flags": ["hallucination_risk", "off_topic",
                           "self_contradictory", "incomplete"],
        # note: groundedness=0.49 < 0.5 but NOT < 0.3, so ungrounded not raised
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot.flags import detect_flags

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct_cases = 0
    details = []

    for case in FLAG_TEST_CASES:
        s = case["scores"]
        actual: list[str] = detect_flags(
            groundedness=s["groundedness"],
            completeness=s["completeness"],
            relevance=s["relevance"],
            consistency=s["consistency"],
            confidence=s["confidence"],
        )
        expected: set[str] = set(case["expected_flags"])
        actual_set: set[str] = set(actual)

        tp = len(expected & actual_set)
        fp = len(actual_set - expected)
        fn = len(expected - actual_set)

        true_positives += tp
        false_positives += fp
        false_negatives += fn
        if expected == actual_set:
            correct_cases += 1

        details.append({
            "label": case["label"],
            "scores": s,
            "expected": sorted(expected),
            "actual": sorted(actual_set),
            "correct": expected == actual_set,
            "tp": tp, "fp": fp, "fn": fn,
        })

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = correct_cases / len(FLAG_TEST_CASES)
    passed = precision > 0.90 and recall > 0.90

    results = {
        "benchmark": "flag_accuracy",
        "test_cases": len(FLAG_TEST_CASES),
        "correct_cases": correct_cases,
        "case_accuracy": round(accuracy, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "targets": {"precision": "> 0.90", "recall": "> 0.90"},
        "passed": passed,
        "details": details,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Flag Detection Accuracy ===")
    print(f"Case accuracy: {accuracy:.1%} ({correct_cases}/{len(FLAG_TEST_CASES)})")
    print(f"Precision:     {precision:.4f}  (target: > 0.90)")
    print(f"Recall:        {recall:.4f}  (target: > 0.90)")
    print(f"F1:            {f1:.4f}")
    print(f"TP={true_positives}  FP={false_positives}  FN={false_negatives}")

    if not passed:
        print("\nFailing cases:")
        for d in details:
            if not d["correct"]:
                print(f"  [{d['label']}]  expected={d['expected']}  "
                      f"got={d['actual']}")

    print(f"Passed:  {'YES ✓' if passed else 'NO ✗'}")
    print(f"Results → {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["passed"] else 1)
