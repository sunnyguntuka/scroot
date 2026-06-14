"""
Benchmark: Confidence metric accuracy.

Tests whether score_confidence() correctly distinguishes highly assertive
responses from heavily hedged ones across a labeled dataset.

Target: Spearman ρ > 0.85 between confidence scores and human labels.

No model loading required - pure regex.

Usage:
    python benchmarks/bench_confidence_accuracy.py

Output:
    benchmarks/results/confidence_accuracy.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "confidence_accuracy.json"

# ---------------------------------------------------------------------------
# Labeled dataset: (response, expected_label)
# label: 0.0 = fully hedged, 0.5 = neutral, 1.0 = fully assertive
# ---------------------------------------------------------------------------

CONFIDENCE_CASES = [
    # --- Highly assertive (label ~0.9-1.0) ---
    {
        "response": "The product is definitely in stock and always ships within 24 hours. Delivery is guaranteed.",
        "label": 1.0,
        "note": "strong assertion markers: definitely, always, guaranteed",
    },
    {
        "response": "This is clearly the best solution available. It absolutely solves the problem without any doubt.",
        "label": 1.0,
        "note": "clearly, absolutely, without doubt",
    },
    {
        "response": "The policy states that refunds must be requested within 30 days. No exceptions.",
        "label": 0.9,
        "note": "must, no exceptions = authoritative assertion",
    },
    {
        "response": "In fact, this feature has never failed in production. It is undoubtedly the most reliable option.",
        "label": 0.95,
        "note": "in fact, never, undoubtedly",
    },
    {
        "response": "You will receive your order by Friday. The tracking number is confirmed.",
        "label": 0.85,
        "note": "will, confirmed = strong commitment",
    },

    # --- Neutral (label ~0.45-0.55) ---
    {
        "response": "The capital of France is Paris. The city has a population of approximately 2 million.",
        "label": 0.5,
        "note": "factual with 'approximately' = slight hedge, otherwise neutral",
    },
    {
        "response": "The function returns a list of strings.",
        "label": 0.5,
        "note": "plain factual statement, no markers",
    },
    {
        "response": "Water boils at 100 degrees Celsius at sea level.",
        "label": 0.5,
        "note": "objective fact, no markers",
    },
    {
        "response": "The meeting is scheduled for 3 PM on Tuesday.",
        "label": 0.5,
        "note": "neutral statement of fact",
    },

    # --- Heavily hedged (label ~0.0-0.2) ---
    {
        "response": "I think the product might be available, but I'm not entirely sure about shipping times. It could take a few days, perhaps longer.",
        "label": 0.0,
        "note": "think, might, not sure, could, perhaps",
    },
    {
        "response": "I believe this may be the right approach, but I'm not certain. It's possible there are better alternatives.",
        "label": 0.0,
        "note": "believe, may, not certain, possible",
    },
    {
        "response": "It seems like the issue could be related to the configuration, though I'm not fully certain about that.",
        "label": 0.05,
        "note": "seems, could, not fully certain",
    },
    {
        "response": "Probably around 500MB, but it roughly depends on your usage. It might be more or less.",
        "label": 0.05,
        "note": "probably, roughly, might",
    },
    {
        "response": "As far as I know, this feature should work, but I'd recommend double-checking as I'm not 100% certain.",
        "label": 0.0,
        "note": "as far as I know, not certain",
    },

    # --- Mixed (label ~0.3-0.7) ---
    {
        "response": "The feature definitely exists, though it might not be available in all plans.",
        "label": 0.5,
        "note": "definitely (assert) + might (hedge) = balanced",
    },
    {
        "response": "This is clearly a bug. It might be fixed in the next release.",
        "label": 0.6,
        "note": "clearly (assert) outweighs might (hedge)",
    },
    {
        "response": "The approach probably works, but you must test it in your environment first.",
        "label": 0.5,
        "note": "probably (hedge) + must (assert) = balanced",
    },
]


# ---------------------------------------------------------------------------
# Spearman ρ
# ---------------------------------------------------------------------------

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


TARGET_RHO = 0.85

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot.metrics.confidence import score_confidence

    scored_labels = []
    scored_scores = []
    details = []

    for case in CONFIDENCE_CASES:
        score, det = score_confidence(case["response"])
        scored_scores.append(score)
        scored_labels.append(case["label"])
        details.append({
            "response": case["response"][:80] + "...",
            "expected_label": case["label"],
            "confidence_score": round(score, 4),
            "delta": round(abs(score - case["label"]), 4),
            "note": case.get("note", ""),
        })

    rho = _spearman(scored_labels, scored_scores)
    mae = sum(abs(s - l) for s, l in zip(scored_scores, scored_labels)) / len(scored_scores)
    passed = rho >= TARGET_RHO

    results = {
        "benchmark": "confidence_accuracy",
        "n_cases": len(CONFIDENCE_CASES),
        "spearman_rho": round(rho, 4),
        "mae": round(mae, 4),
        "target_rho": TARGET_RHO,
        "passed": passed,
        "details": details,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Confidence Metric Accuracy ===")
    print(f"  N cases:      {len(CONFIDENCE_CASES)}")
    print(f"  Spearman ρ:   {rho:+.4f}  (target: >= {TARGET_RHO})")
    print(f"  MAE:          {mae:.4f}")
    print(f"  Passed:       {'YES ✓' if passed else 'NO ✗'}")
    print(f"  Results →     {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["passed"] else 1)
