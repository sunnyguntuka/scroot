"""
Benchmark: Claim extraction precision and recall.

Uses hand-labeled test cases where the expected factual claims are known.
Measures how accurately extract_claims() identifies real factual claims
(precision) versus missing them (recall).

Target: Precision > 0.85, Recall > 0.80

No model loading required - runs in < 1 second.

Usage:
    python benchmarks/bench_claim_accuracy.py

Output:
    benchmarks/results/claim_accuracy.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "claim_accuracy.json"

# ---------------------------------------------------------------------------
# Hand-labeled test cases
# Each case specifies what extract_claims() SHOULD and SHOULD NOT extract.
# ---------------------------------------------------------------------------

CLAIM_TEST_CASES = [
    {
        "response": (
            "We offer a 30-day full refund at no extra cost. "
            "You can return any item within 30 days of purchase. "
            "Hi there!"
        ),
        "expected_claims": [
            "We offer a 30-day full refund at no extra cost.",
            "You can return any item within 30 days of purchase.",
        ],
        "expected_non_claims": ["Hi there!"],
    },
    {
        "response": (
            "What is your name? "
            "My name is Claude. "
            "I am an AI assistant built by Anthropic."
        ),
        "expected_claims": [
            "My name is Claude.",
            "I am an AI assistant built by Anthropic.",
        ],
        "expected_non_claims": ["What is your name?"],
    },
    {
        "response": (
            "Sure! "
            "The capital of France is Paris. "
            "It has a population of approximately 2.1 million people."
        ),
        "expected_claims": [
            "The capital of France is Paris.",
            "It has a population of approximately 2.1 million people.",
        ],
        "expected_non_claims": ["Sure!"],
    },
    {
        "response": (
            "Thanks for asking. "
            "Yes. "
            "No. "
            "The product ships within 3-5 business days to all US addresses."
        ),
        "expected_claims": [
            "The product ships within 3-5 business days to all US addresses.",
        ],
        "expected_non_claims": ["Thanks for asking.", "Yes.", "No."],
    },
    {
        "response": (
            "Okay! "
            "The speed of light is approximately 299,792,458 metres per second. "
            "This constant is denoted by the symbol c in physics equations. "
            "Certainly, that's the foundation of special relativity."
        ),
        "expected_claims": [
            "The speed of light is approximately 299,792,458 metres per second.",
            "This constant is denoted by the symbol c in physics equations.",
        ],
        "expected_non_claims": ["Okay!", "Certainly, that's the foundation of special relativity."],
    },
    {
        "response": (
            "Water freezes at 0 degrees Celsius at standard atmospheric pressure. "
            "It boils at 100 degrees Celsius under the same conditions."
        ),
        "expected_claims": [
            "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
            "It boils at 100 degrees Celsius under the same conditions.",
        ],
        "expected_non_claims": [],
    },
    {
        "response": "Hello! How are you today? Can I help you with something?",
        "expected_claims": [],
        "expected_non_claims": [
            "Hello!",
            "How are you today?",
            "Can I help you with something?",
        ],
    },
    {
        "response": (
            "The Eiffel Tower was completed in 1889. "
            "It stands 330 metres tall including its broadcast antenna. "
            "It was designed by Gustave Eiffel."
        ),
        "expected_claims": [
            "The Eiffel Tower was completed in 1889.",
            "It stands 330 metres tall including its broadcast antenna.",
            "It was designed by Gustave Eiffel.",
        ],
        "expected_non_claims": [],
    },
]


# ---------------------------------------------------------------------------
# Matching helper - fuzzy substring matching to handle minor tokenisation diffs
# ---------------------------------------------------------------------------

def _is_match(extracted: str, expected: str, threshold: float = 0.7) -> bool:
    """Return True if extracted substantially overlaps with expected."""
    e = extracted.lower().strip().rstrip(".")
    x = expected.lower().strip().rstrip(".")
    # Exact or substring match
    if e in x or x in e:
        return True
    # Word-overlap (Jaccard-like)
    e_words = set(e.split())
    x_words = set(x.split())
    if not e_words or not x_words:
        return False
    overlap = len(e_words & x_words) / len(e_words | x_words)
    return overlap >= threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot.text_utils import extract_claims

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    details = []

    for case in CLAIM_TEST_CASES:
        extracted: list[str] = extract_claims(case["response"])
        expected: list[str] = case["expected_claims"]
        non_claims: list[str] = case["expected_non_claims"]

        # Recall: each expected claim should be matched by at least one extracted
        case_tp = 0
        case_fn = 0
        for exp in expected:
            if any(_is_match(ext, exp) for ext in extracted):
                case_tp += 1
            else:
                case_fn += 1

        # Precision: each extracted claim should match at least one expected
        case_fp = 0
        for ext in extracted:
            is_valid = any(_is_match(ext, exp) for exp in expected)
            if not is_valid:
                case_fp += 1

        true_positives += case_tp
        false_positives += case_fp
        false_negatives += case_fn

        details.append({
            "response": case["response"][:100] + "...",
            "expected_claims": len(expected),
            "extracted_count": len(extracted),
            "extracted": extracted,
            "tp": case_tp, "fp": case_fp, "fn": case_fn,
        })

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    passed = precision > 0.85 and recall > 0.80

    results = {
        "benchmark": "claim_accuracy",
        "test_cases": len(CLAIM_TEST_CASES),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "targets": {"precision": "> 0.85", "recall": "> 0.80"},
        "passed": passed,
        "details": details,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Claim Extraction Accuracy ===")
    print(f"Precision: {precision:.4f}  (target: > 0.85)")
    print(f"Recall:    {recall:.4f}  (target: > 0.80)")
    print(f"F1:        {f1:.4f}")
    print(f"TP={true_positives}  FP={false_positives}  FN={false_negatives}")
    print(f"Passed:    {'YES ✓' if passed else 'NO ✗'}")
    print(f"Results → {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    run()
    sys.exit(0 if run()["passed"] else 1)
