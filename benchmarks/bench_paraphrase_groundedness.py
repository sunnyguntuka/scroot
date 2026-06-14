"""
Benchmark: Groundedness on paraphrased responses.

Tests whether score_groundedness() correctly identifies responses that
paraphrase the context (should be grounded) vs those that contradict it
or introduce new facts (should be ungrounded).

This validates the bi-encoder similarity fallback - cases where exact NLI
entailment would fail on paraphrases but semantic similarity should succeed.

Target: Accuracy > 0.80 (fraction of cases correctly classified).

Usage:
    python benchmarks/bench_paraphrase_groundedness.py

Output:
    benchmarks/results/paraphrase_groundedness.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "paraphrase_groundedness.json"

# ---------------------------------------------------------------------------
# Labeled dataset
# grounded: True = response is supported by context, False = not supported
# ---------------------------------------------------------------------------

PARAPHRASE_CASES = [
    # --- Exact match (trivially grounded) ---
    {
        "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
        "response": "All customers are eligible for a 30-day full refund at no extra cost.",
        "grounded": True,
        "type": "exact",
        "note": "verbatim from context",
    },

    # --- Paraphrases (should be grounded via similarity fallback) ---
    {
        "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
        "response": "We offer a 30-day money-back guarantee to all our customers for free.",
        "grounded": True,
        "type": "paraphrase",
        "note": "paraphrase of refund policy",
    },
    {
        "context": ["The product ships within 5 business days to all US addresses."],
        "response": "Delivery takes approximately one week within the United States.",
        "grounded": True,
        "type": "paraphrase",
        "note": "5 business days ≈ one week, paraphrase",
    },
    {
        "context": ["Our support team is available Monday through Friday, 9 AM to 6 PM EST."],
        "response": "Customer support operates on weekdays during business hours Eastern time.",
        "grounded": True,
        "type": "paraphrase",
        "note": "weekdays during business hours ≈ Mon-Fri 9-6 EST",
    },
    {
        "context": ["The software requires a minimum of 8 GB of RAM and 2 GB of disk space."],
        "response": "You need at least 8 gigabytes of memory and 2 gigabytes of storage to run this.",
        "grounded": True,
        "type": "paraphrase",
        "note": "RAM→memory, disk space→storage",
    },
    {
        "context": ["Premium members receive a 20% discount on all purchases."],
        "response": "If you are a premium subscriber, you get one-fifth off everything you buy.",
        "grounded": True,
        "type": "paraphrase",
        "note": "20% = one-fifth, paraphrase",
    },

    # --- Contradictions (should be NOT grounded) ---
    {
        "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
        "response": "We offer a 90-day money-back guarantee with a 15% restocking fee.",
        "grounded": False,
        "type": "contradiction",
        "note": "wrong duration (90 vs 30) and adds restocking fee",
    },
    {
        "context": ["The product ships within 5 business days to all US addresses."],
        "response": "International shipping takes up to 3 weeks.",
        "grounded": False,
        "type": "contradiction",
        "note": "wrong scope (international vs US) and timeframe",
    },
    {
        "context": ["Our support team is available Monday through Friday, 9 AM to 6 PM EST."],
        "response": "Customer support is available 24 hours a day, 7 days a week.",
        "grounded": False,
        "type": "contradiction",
        "note": "24/7 contradicts Mon-Fri 9-6",
    },

    # --- Hallucinations (new facts not in context) ---
    {
        "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
        "response": "Premium customers get a 60-day refund while standard customers get 30 days.",
        "grounded": False,
        "type": "hallucination",
        "note": "introduces premium/standard distinction not in context",
    },
    {
        "context": ["The software requires a minimum of 8 GB of RAM."],
        "response": "The software requires 8 GB of RAM and a dedicated GPU with 4 GB VRAM.",
        "grounded": False,
        "type": "hallucination",
        "note": "adds GPU requirement not mentioned in context",
    },

    # --- Off-topic (completely unrelated) ---
    {
        "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
        "response": "The Eiffel Tower was built in 1889 and stands 330 metres tall.",
        "grounded": False,
        "type": "off_topic",
        "note": "completely unrelated to refund policy",
    },
    {
        "context": ["Premium members receive a 20% discount on all purchases."],
        "response": "Photosynthesis converts sunlight into glucose using chlorophyll.",
        "grounded": False,
        "type": "off_topic",
        "note": "completely unrelated to discount policy",
    },
]

TARGET_ACCURACY = 0.80


def run() -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot.metrics.groundedness import score_groundedness

    correct = 0
    details = []

    for case in PARAPHRASE_CASES:
        score, det = score_groundedness(
            response=case["response"],
            context=case["context"],
            embedding_model="all-MiniLM-L6-v2",
        )

        predicted_grounded = score >= 0.5
        is_correct = predicted_grounded == case["grounded"]
        if is_correct:
            correct += 1

        details.append({
            "type": case["type"],
            "response": case["response"][:70] + "...",
            "expected_grounded": case["grounded"],
            "groundedness_score": round(score, 4),
            "predicted_grounded": predicted_grounded,
            "correct": is_correct,
            "note": case.get("note", ""),
        })

    accuracy = correct / len(PARAPHRASE_CASES)
    passed = accuracy >= TARGET_ACCURACY

    # Breakdown by type
    type_breakdown: dict[str, dict] = {}
    for d in details:
        t = d["type"]
        if t not in type_breakdown:
            type_breakdown[t] = {"total": 0, "correct": 0}
        type_breakdown[t]["total"] += 1
        if d["correct"]:
            type_breakdown[t]["correct"] += 1

    results = {
        "benchmark": "paraphrase_groundedness",
        "n_cases": len(PARAPHRASE_CASES),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "target_accuracy": TARGET_ACCURACY,
        "passed": passed,
        "by_type": {
            t: {
                "accuracy": round(v["correct"] / v["total"], 3),
                "correct": v["correct"],
                "total": v["total"],
            }
            for t, v in type_breakdown.items()
        },
        "details": details,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Paraphrase Groundedness Accuracy ===")
    print(f"  N cases:       {len(PARAPHRASE_CASES)}")
    print(f"  Accuracy:      {accuracy:.1%}  ({correct}/{len(PARAPHRASE_CASES)})")
    print(f"  Target:        >= {TARGET_ACCURACY:.0%}")
    print(f"  Passed:        {'YES ✓' if passed else 'NO ✗'}")
    print(f"\n  By type:")
    for t, v in results["by_type"].items():
        print(f"    {t:<15} {v['accuracy']:.0%}  ({v['correct']}/{v['total']})")

    if not passed:
        print(f"\n  Failures:")
        for d in details:
            if not d["correct"]:
                print(f"    [{d['type']}] score={d['groundedness_score']:.3f}"
                      f"  expected={'grounded' if d['expected_grounded'] else 'ungrounded'}"
                      f"  ({d['note']})")

    print(f"\n  Results → {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["passed"] else 1)
