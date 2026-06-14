"""
Prepare Google Natural Questions (NQ) benchmark dataset.

Downloads the NQ validation split from HuggingFace (streaming - no full
download required) and extracts 500 examples with valid short answers and
long-answer context passages.

Generates five perturbation levels per example:

  A0 - Correct, fully grounded answer (IQS target: ≥ 0.75)
  A1 - Correct answer with added hedging/verbosity (IQS target: ≥ 0.65)
  A2 - Partially hallucinated: one injected wrong fact (IQS target: 0.35–0.55)
  A3 - Related but non-answering sentence from context (IQS target: 0.20–0.40)
  A4 - Completely off-topic response (IQS target: ≤ 0.20)

No LLM calls. All perturbations are rule-based and seeded for reproducibility.

Usage:
    python benchmarks/datasets/prepare_nq.py
    python benchmarks/datasets/prepare_nq.py --n 100 --seed 42
    python benchmarks/datasets/prepare_nq.py --output custom.jsonl

Output: benchmarks/datasets/nq_500.jsonl
Each line:
    {
      "id": str,
      "question": str,
      "context": str,           # Long-answer passage (≤ 1 500 chars)
      "answer": str,            # Gold short answer
      "perturbations": {
        "A0": {"response": str, "level": 0},
        ...
        "A4": {"response": str, "level": 4}
      }
    }
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Off-topic templates for A4 (completely unrelated to the question)
# ---------------------------------------------------------------------------

_OFFTOPIC = [
    "The mitochondria is the powerhouse of the cell, producing ATP through "
    "oxidative phosphorylation during cellular respiration.",
    "Photosynthesis occurs in chloroplasts, where sunlight converts carbon "
    "dioxide and water into glucose and oxygen.",
    "The water cycle describes the continuous movement of water through "
    "evaporation, condensation, precipitation, and collection.",
    "Supply and demand curves intersect at the equilibrium price in a "
    "competitive free-market economy.",
    "The Pythagorean theorem states that a² + b² = c² for any right triangle, "
    "where c is the length of the hypotenuse.",
    "Newton's first law of motion holds that an object at rest stays at rest "
    "and an object in motion stays in motion unless acted upon by a net force.",
    "Plate tectonics describes the movement of the Earth's lithospheric plates "
    "driven by convection currents in the mantle.",
    "The speed of light in a vacuum is approximately 299 792 458 metres per "
    "second, denoted by the symbol c.",
    "DNA replication is a semiconservative process in which each strand of the "
    "double helix serves as a template for the new complementary strand.",
    "Inflation is measured by tracking the price changes in a representative "
    "basket of goods and services over time.",
]

# Wrong replacements for number-swap perturbation (A2)
_NUMBER_OFFSETS = [3, -3, 7, -7, 11, -11, 13, -13]

# Hedging phrases for A1
_HEDGE_PREFIXES = [
    "According to available sources, ",
    "Based on the information provided, ",
    "It is generally understood that ",
    "Historical records suggest that ",
    "As widely documented, ",
]

_HEDGE_SUFFIXES = [
    ", though this may vary depending on the specific context.",
    ", although some sources may offer differing interpretations.",
    ", as supported by the surrounding documentation.",
    ", according to the most commonly cited references.",
    ", based on the evidence available at the time.",
]


# ---------------------------------------------------------------------------
# Perturbation generators
# ---------------------------------------------------------------------------

def _make_a0(answer: str, question: str) -> str:
    """Correct, grounded response. Wrap the answer in a natural sentence."""
    q = question.rstrip("?").strip()
    # Use a simple template that states the answer as a fact
    return f"The answer to '{q}' is {answer}."


def _make_a1(answer: str, rng: random.Random) -> str:
    """Correct but verbose / hedged response."""
    prefix = rng.choice(_HEDGE_PREFIXES)
    suffix = rng.choice(_HEDGE_SUFFIXES)
    return f"{prefix}{answer}{suffix}"


def _make_a2(answer: str, rng: random.Random) -> str:
    """Partially hallucinated: swap one number, or inject a wrong adjective."""
    # Strategy 1: replace the first integer with a nearby wrong value
    numbers = list(re.finditer(r"\b(\d{2,4})\b", answer))
    if numbers:
        match = numbers[0]
        original = int(match.group(1))
        offset = rng.choice(_NUMBER_OFFSETS)
        wrong = str(abs(original + offset))
        modified = answer[: match.start()] + wrong + answer[match.end() :]
        return f"Based on the available information, {modified}."

    # Strategy 2: replace a capitalised word with a wrong alternative
    caps = list(re.finditer(r"\b([A-Z][a-z]{3,})\b", answer))
    wrong_words = ["American", "British", "French", "German", "European",
                   "Northern", "Southern", "Eastern", "Western", "Ancient"]
    if caps:
        match = rng.choice(caps)
        replacement = rng.choice([w for w in wrong_words
                                   if w.lower() != match.group(1).lower()])
        modified = answer[: match.start()] + replacement + answer[match.end():]
        return f"Based on the available information, {modified}."

    # Strategy 3: append a plausible-sounding wrong date
    wrong_year = rng.choice(["1847", "1923", "1965", "2003", "1776"])
    return f"{answer}, which occurred in {wrong_year}."


def _make_a3(context: str, answer: str, rng: random.Random) -> str:
    """Related but non-answering: random sentence from context that lacks answer."""
    # Split context into sentences
    sents = re.split(r"(?<=[.!?])\s+", context)
    sents = [s.strip() for s in sents if len(s.strip().split()) >= 6]

    # Prefer sentences that don't contain key words from the answer
    answer_words = {w.lower() for w in answer.split() if len(w) > 3}
    non_answer = [
        s for s in sents
        if not any(w in s.lower() for w in answer_words)
    ]

    candidates = non_answer if non_answer else sents
    if candidates:
        return rng.choice(candidates)

    # Fallback: extract the first 120 chars of context
    return context[:120].strip() + "..."


def _make_a4(rng: random.Random) -> str:
    """Completely off-topic response."""
    return rng.choice(_OFFTOPIC)


def generate_perturbations(
    question: str,
    context: str,
    answer: str,
    rng: random.Random,
) -> dict[str, dict]:
    return {
        "A0": {"response": _make_a0(answer, question), "level": 0},
        "A1": {"response": _make_a1(answer, rng), "level": 1},
        "A2": {"response": _make_a2(answer, rng), "level": 2},
        "A3": {"response": _make_a3(context, answer, rng), "level": 3},
        "A4": {"response": _make_a4(rng), "level": 4},
    }


# ---------------------------------------------------------------------------
# NQ extraction helpers
# ---------------------------------------------------------------------------

def _tokens_to_text(tokens: list[dict], start: int, end: int) -> str:
    """Reconstruct clean text from NQ token slice."""
    chunk = tokens[start:end]
    words = [t["token"] for t in chunk if not t.get("is_html", False)]
    text = " ".join(words)
    # Collapse whitespace and wiki markup artifacts
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)   # citation numbers [1]
    text = re.sub(r"<[^>]+>", "", text)           # any stray HTML
    return text.strip()


def _extract_example(raw: dict) -> dict | None:
    """Return {id, question, context, answer} or None if unusable."""
    question = raw["question"]["text"].strip()
    tokens = raw["document"]["tokens"]

    for ann in raw["annotations"]:
        short_answers = ann.get("short_answers", [])
        if not short_answers:
            continue

        sa = short_answers[0]
        sa_start = sa["start_token"]
        sa_end = sa["end_token"]

        # Try the precomputed text field first (present in some HF versions)
        answer = ""
        if sa.get("text"):
            raw_text = sa["text"]
            answer = (
                " ".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
            ).strip()
        if not answer and sa_start >= 0 and sa_end > sa_start:
            answer = _tokens_to_text(tokens, sa_start, sa_end)

        if not answer or len(answer.split()) < 1:
            continue

        # Extract long-answer passage for context
        la = ann.get("long_answer", {})
        la_start = la.get("start_token", -1)
        la_end = la.get("end_token", -1)
        if la_start < 0 or la_end <= la_start:
            # No long answer - use surrounding tokens
            lo = max(0, sa_start - 150)
            hi = min(len(tokens), sa_end + 150)
            context = _tokens_to_text(tokens, lo, hi)
        else:
            context = _tokens_to_text(tokens, la_start, la_end)

        # Cap context length to keep scoring tractable
        context = context[:1500]

        if len(context.split()) < 20:
            continue

        return {
            "id": str(raw.get("id", "")),
            "question": question,
            "context": context,
            "answer": answer,
        }

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(n: int = 500, seed: int = 42, output: str | None = None) -> Path:
    output_path = Path(output) if output else (
        Path(__file__).parent / "nq_500.jsonl"
    )

    if output_path.exists():
        existing = sum(1 for _ in output_path.open())
        if existing >= n:
            print(f"Dataset already exists at {output_path} ({existing} examples). "
                  f"Delete it to regenerate.")
            return output_path

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Install with:\n"
              "  pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        from tqdm import tqdm
        progress = tqdm(total=n, desc="Extracting NQ examples")
    except ImportError:
        class _FakePbar:
            def update(self, n=1): pass
            def close(self): pass
            def write(self, s): print(s)
        progress = _FakePbar()

    print(f"Streaming NQ validation split from HuggingFace...")
    ds = load_dataset(
        "natural_questions",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    rng = random.Random(seed)
    examples = []
    scanned = 0

    for raw in ds:
        scanned += 1
        ex = _extract_example(raw)
        if ex is None:
            continue

        ex["perturbations"] = generate_perturbations(
            ex["question"], ex["context"], ex["answer"], rng
        )
        examples.append(ex)
        progress.update(1)

        if len(examples) >= n:
            break

        if scanned % 500 == 0:
            progress.write(
                f"  Scanned {scanned} raw items, kept {len(examples)}"
            )

    progress.close()

    if len(examples) < n:
        print(f"Warning: only found {len(examples)} valid examples "
              f"(target was {n}). Saving what we have.", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(examples)} examples → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500,
                        help="Number of examples to extract (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for perturbation generation (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: benchmarks/datasets/nq_500.jsonl)")
    args = parser.parse_args()
    prepare(n=args.n, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
