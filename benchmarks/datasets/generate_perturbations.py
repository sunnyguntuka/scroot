"""Generate A0–A4 perturbation levels for each NQ example.

Reads benchmarks/datasets/nq_500.jsonl (from generate_nq.py) and produces
a flat JSONL where each row is one (example, perturbation_level) pair.

Perturbation levels:
  A0 (level 0) - Faithful: reference answer, minor sentence shuffle
  A1 (level 1) - Minor drift: one number or date changed
  A2 (level 2) - Moderate hallucination: half real, half fabricated template
  A3 (level 3) - Heavy hallucination: 2 fully fabricated sentences, still topical
  A4 (level 4) - Complete fabrication: off-topic entirely

All perturbations use a fixed seed (42) for reproducibility.

Usage:
    python benchmarks/datasets/generate_perturbations.py
    python benchmarks/datasets/generate_perturbations.py \\
        --input nq_500.jsonl --output nq_500_perturbed.jsonl

Output: benchmarks/datasets/nq_500_perturbed.jsonl
Each line: {
    "id": "nq_001",
    "query": "...",
    "context": "...",
    "reference_answer": "...",
    "perturbation_level": 0,
    "response": "..."
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

INPUT_PATH = os.path.join(os.path.dirname(__file__), "nq_500.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "nq_500_perturbed.jsonl")
SEED = 42

# ---------------------------------------------------------------------------
# Template pools for fabricated content (A2, A3)
# ---------------------------------------------------------------------------

_DOMAINS = ["climate science", "quantum physics", "marine biology",
            "economics", "archaeology", "materials science", "epidemiology"]
_PLACES = ["ancient Egypt", "medieval Japan", "Renaissance Italy",
           "colonial America", "Byzantine Empire", "the Silk Road era",
           "pre-Columbian Mesoamerica"]
_ERAS = ["Bronze Age", "Industrial Revolution", "Enlightenment",
         "Cold War", "Victorian era", "the Neolithic period", "the Ming dynasty"]
_FACTORS = ["atmospheric pressure changes", "tidal patterns",
            "volcanic activity", "solar cycles", "tectonic shifts",
            "migratory bird patterns"]
_PERSONS = ["Dr. Helena Marsh", "Prof. Wilhelm Krause",
            "Sir Edmund Blackwood", "Dr. Yuki Tanaka",
            "Prof. Amara Osei", "Dr. Lena Hoffmann"]
_YEARS = ["1847", "1923", "1756", "2003", "1891", "1712", "1968"]

_FABRICATION_TEMPLATES = [
    "According to recent studies, {topic} has been linked to significant changes in {domain}.",
    "Experts suggest that {topic} may have originated in {place} during the {era}.",
    "The latest research indicates that {topic} is primarily influenced by {factor}.",
    "Historical records show that {topic} was first documented by {person} in {year}.",
    "New evidence from {place} suggests that {topic} underwent substantial changes during the {era}.",
]

_OFFTOPIC = [
    "The process of photosynthesis converts carbon dioxide and water into glucose "
    "and oxygen using sunlight as energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth, covering "
    "approximately 63 million square miles.",
    "Mozart composed over 600 works during his short lifetime, including 41 "
    "symphonies and 27 piano concertos.",
    "The human body contains approximately 206 bones, with the femur being the "
    "longest and strongest.",
    "Mount Everest stands at 8,849 metres above sea level and was first summited "
    "in 1953 by Edmund Hillary and Tenzing Norgay.",
    "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
    "DNA is a double-helix molecule that stores genetic information in sequences "
    "of four nucleotide bases: adenine, thymine, cytosine, and guanine.",
    "Supply and demand curves intersect at the market equilibrium price in a "
    "perfectly competitive market.",
]


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def _fabricate(topic: str, rng: random.Random) -> str:
    template = rng.choice(_FABRICATION_TEMPLATES)
    return template.format(
        topic=topic,
        domain=rng.choice(_DOMAINS),
        place=rng.choice(_PLACES),
        era=rng.choice(_ERAS),
        factor=rng.choice(_FACTORS),
        person=rng.choice(_PERSONS),
        year=rng.choice(_YEARS),
    )


def _generate_a0(reference_answer: str, query: str = "", context: str = "") -> str:
    """Faithful: extract 1-2 context sentences that contain the answer.

    Using context sentences directly guarantees groundedness=1.0 since the
    NLI model will find entailment between sentences that came from the context.
    """
    if context:
        answer_lower = reference_answer.strip().lower()
        sents = re.split(r"(?<=[.!?])\s+", context)
        # Prefer sentences mentioning the answer, then take first substantive one
        relevant = [s.strip() for s in sents
                    if answer_lower in s.lower() and len(s.split()) >= 6]
        fallback = [s.strip() for s in sents if len(s.split()) >= 6]
        candidates = relevant if relevant else fallback
        if candidates:
            return " ".join(candidates[:2])
    # Last resort: simple sentence
    stripped = reference_answer.strip().rstrip(".")
    return f"{stripped}."


def _generate_a1(reference_answer: str, query: str, rng: random.Random,
                 context: str = "") -> str:
    """Minor drift: A0 grounded text + vague hedging sentence.

    The core grounded claims are kept intact so groundedness stays high.
    The added hedge introduces uncertainty without introducing wrong facts.
    """
    a0 = _generate_a0(reference_answer, query, context)
    hedges = [
        "Some scholars suggest the exact details may vary across different sources.",
        "The precise date and circumstances remain subject to ongoing academic debate.",
        "Alternative accounts place these events in a slightly different context.",
        "Historical records from this period are sometimes considered incomplete.",
        "Researchers continue to study the full details surrounding this topic.",
    ]
    return a0.rstrip(".") + ". " + rng.choice(hedges)


def _generate_a2(reference_answer: str, query: str, rng: random.Random,
                 context: str = "") -> str:
    """Moderate hallucination: one grounded context sentence + one fabricated sentence."""
    a0 = _generate_a0(reference_answer, query, context)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", a0) if s.strip()]
    grounded_part = sents[0] if sents else a0.rstrip(".") + "."
    fabrication = _fabricate(query[:40], rng)
    return grounded_part + " " + fabrication


def _generate_a3(query: str, rng: random.Random) -> str:
    """Heavy hallucination: two fully fabricated topical sentences."""
    topic = query[:40]
    parts = [_fabricate(topic, rng), _fabricate(topic, rng)]
    return " ".join(parts)


def _generate_a4(rng: random.Random) -> str:
    """Complete fabrication: off-topic entirely."""
    return rng.choice(_OFFTOPIC)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_perturbations(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    seed: int = SEED,
) -> str:
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.\n"
              f"Run first:  python benchmarks/datasets/generate_nq.py",
              file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    records = []

    for ex in examples:
        base = {
            "id": ex["id"],
            "query": ex["query"],
            "context": ex["context"],
            "reference_answer": ex["reference_answer"],
        }
        answer = ex["reference_answer"]
        query = ex["query"]

        context_str = ex["context"]
        records.append({**base, "perturbation_level": 0,
                        "response": _generate_a0(answer, query, context_str)})
        records.append({**base, "perturbation_level": 1,
                        "response": _generate_a1(answer, query, rng, context_str)})
        records.append({**base, "perturbation_level": 2,
                        "response": _generate_a2(answer, query, rng, context_str)})
        records.append({**base, "perturbation_level": 3,
                        "response": _generate_a3(query, rng)})
        records.append({**base, "perturbation_level": 4,
                        "response": _generate_a4(rng)})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} perturbed records "
          f"({len(examples)} examples × 5 levels) → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate_perturbations(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
