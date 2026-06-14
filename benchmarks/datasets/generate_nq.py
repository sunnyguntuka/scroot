"""Download and prepare a 500-example subset of Google Natural Questions.

Google NQ contains real user queries from Google Search with human-annotated
answers extracted from Wikipedia pages.

Source: https://huggingface.co/datasets/google-research-datasets/natural_questions

For each example, extracts:
  - query:            the user's question
  - context:          the Wikipedia paragraph containing the answer
  - reference_answer: the human-annotated short answer

Output: benchmarks/datasets/nq_500.jsonl
Each line: {"id": "nq_001", "query": "...", "context": "...", "reference_answer": "..."}

Run this before generate_perturbations.py:
    python benchmarks/datasets/generate_nq.py
    python benchmarks/datasets/generate_nq.py --n 100  # smaller test set
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "nq_500.jsonl")
SEED = 42
SAMPLE_SIZE = 500
MIN_ANSWER_WORDS = 2
MIN_CONTEXT_WORDS = 20
CONTEXT_WINDOW = 75   # tokens on each side of answer span


def _tokens_to_text(token_list: list, start: int, end: int) -> str:
    """Reconstruct clean text from a NQ token slice.

    Handles both dict-style tokens ({token: str, is_html: bool}) and
    plain-string token lists, since the HuggingFace dataset representation
    varies slightly across versions.
    """
    chunk = token_list[start:end]
    words = []
    for t in chunk:
        if isinstance(t, dict):
            if not t.get("is_html", False):
                words.append(t["token"])
        else:
            words.append(str(t))
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)   # strip wiki citation marks
    return text.strip()


def _extract(item: dict) -> dict | None:
    """Return {query, context, reference_answer} or None if item is unusable.

    Handles the current HuggingFace NQ format where annotations and tokens
    are returned as columnar dicts (arrays of values) rather than lists of dicts.
    """
    question = (item.get("question") or {}).get("text", "").strip()
    if not question:
        return None

    doc = item.get("document") or {}
    tokens_field = doc.get("tokens") or {}

    # Tokens are a dict of parallel arrays: {token: [...], is_html: [...], ...}
    if isinstance(tokens_field, dict):
        token_strs = tokens_field.get("token", [])
        is_html_list = tokens_field.get("is_html", [False] * len(token_strs))
        # Reconstruct as list of dicts for _tokens_to_text compatibility
        token_list = [{"token": t, "is_html": h}
                      for t, h in zip(token_strs, is_html_list)]
    elif isinstance(tokens_field, list):
        token_list = tokens_field
    else:
        return None

    if not token_list:
        return None

    # Annotations: columnar dict {id: [...], short_answers: [...], long_answer: [...]}
    annotations = item.get("annotations") or {}
    if not annotations:
        return None

    # Number of annotators
    ann_ids = annotations.get("id", [])
    n_annotators = len(ann_ids)

    for i in range(n_annotators):
        short_answers_row = (annotations.get("short_answers") or [])[i] if i < len(annotations.get("short_answers") or []) else {}
        if not short_answers_row:
            continue

        # short_answers_row is itself a dict with parallel arrays:
        # {text: [...], start_token: [...], end_token: [...], ...}
        texts = short_answers_row.get("text", [])
        start_tokens = short_answers_row.get("start_token", [])
        end_tokens = short_answers_row.get("end_token", [])

        if not texts:
            continue

        # Use the first short answer
        reference_answer = texts[0].strip() if texts else ""
        sa_start = start_tokens[0] if start_tokens else -1
        sa_end = end_tokens[0] if end_tokens else -1

        if not reference_answer and sa_start >= 0 and sa_end > sa_start:
            reference_answer = _tokens_to_text(token_list, sa_start, sa_end)

        if not reference_answer or len(reference_answer.split()) < MIN_ANSWER_WORDS:
            continue

        # Long answer for context
        long_answers = annotations.get("long_answer") or []
        la = long_answers[i] if i < len(long_answers) else {}
        la_start = la.get("start_token", -1) if isinstance(la, dict) else -1
        la_end = la.get("end_token", -1) if isinstance(la, dict) else -1

        if la_start >= 0 and la_end > la_start:
            context = _tokens_to_text(token_list, la_start, la_end)
        elif sa_start >= 0:
            ctx_start = max(0, sa_start - CONTEXT_WINDOW)
            ctx_end = min(len(token_list), sa_end + CONTEXT_WINDOW)
            context = _tokens_to_text(token_list, ctx_start, ctx_end)
        else:
            continue

        context = context[:1500]
        if len(context.split()) < MIN_CONTEXT_WORDS:
            continue

        return {
            "query": question,
            "context": context,
            "reference_answer": reference_answer,
        }

    return None


def download_and_prepare(
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
    output: str = OUTPUT_PATH,
) -> str:
    if os.path.exists(output):
        count = sum(1 for _ in open(output))
        if count >= n:
            print(f"Dataset already at {output} ({count} examples). "
                  f"Delete to regenerate.")
            return output

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        from tqdm import tqdm
        bar = tqdm(total=n, desc="Extracting NQ examples")
    except ImportError:
        class _Bar:
            def update(self, n=1): pass
            def close(self): pass
            def write(self, s): print(s)
        bar = _Bar()

    print("Streaming NQ validation split from HuggingFace...")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split="validation",
        streaming=True,
    )

    # Collect up to 2× the target so we can random-sample for diversity
    candidates = []
    scanned = 0
    for item in ds:
        scanned += 1
        ex = _extract(item)
        if ex is not None:
            candidates.append(ex)
            bar.update(1)
        if len(candidates) >= n * 2:
            break
        if scanned % 500 == 0:
            bar.write(f"  scanned {scanned}, found {len(candidates)}")

    bar.close()

    rng = random.Random(seed)
    selected = rng.sample(candidates, min(n, len(candidates)))

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(selected):
            ex["id"] = f"nq_{i:03d}"
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} examples → {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    download_and_prepare(n=args.n, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
