"""
Task 3 - groundedness NLI latency benchmark for top-k premise pre-filtering.

`score_groundedness` runs an NLI cross-encoder on every (premise_sentence,
claim) pair. With long contexts this is the dominant cost. The new
`top_k_premises` parameter caps the number of premises per claim by embedding
similarity to the claim BEFORE the NLI step.

This benchmark builds synthetic contexts of 5/10/20/40 sentences, scores a fixed
5-claim response with top-k filtering OFF and ON (k=5), and reports the speedup
and the score delta at each context size.

Run:
  set PYTHONIOENCODING=utf-8
  python benchmarks/bench_groundedness_latency.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from scroot.metrics.groundedness import score_groundedness  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
EMB = "all-MiniLM-L6-v2"

RESPONSE = (
    "The Eiffel Tower is located in Paris. "
    "It was completed in 1889. "
    "The tower is made of wrought iron. "
    "It stands about 330 meters tall. "
    "Millions of tourists visit it every year."
)

# Sentences that genuinely support the claims, padded with distractors so longer
# contexts add noise (the realistic case top-k filtering targets).
_SUPPORT = [
    "The Eiffel Tower is a landmark in Paris, France.",
    "Construction of the Eiffel Tower finished in 1889.",
    "The Eiffel Tower is built from wrought iron.",
    "The Eiffel Tower is roughly 330 meters in height.",
    "Each year millions of visitors come to see the Eiffel Tower.",
]
_DISTRACTORS = [
    "The Louvre museum houses the Mona Lisa painting.",
    "The Seine river flows through the centre of the city.",
    "French cuisine is celebrated around the world.",
    "The TGV is a high-speed rail network in France.",
    "Notre-Dame cathedral is a Gothic masterpiece.",
    "Croissants are a popular French pastry.",
    "The metro system serves the entire metropolitan area.",
    "Many cafes line the historic boulevards.",
    "The country is famous for its vineyards and wine.",
    "Street markets sell fresh produce daily.",
]


def _build_context(n_sentences: int) -> list[str]:
    ctx = list(_SUPPORT)
    i = 0
    while len(ctx) < n_sentences:
        ctx.append(_DISTRACTORS[i % len(_DISTRACTORS)])
        i += 1
    return ctx[:n_sentences]


def _time_score(context, top_k_premises, repeats=3):
    best = None
    score = None
    for _ in range(repeats):
        t = time.perf_counter()
        score, _ = score_groundedness(
            RESPONSE, context,
            nli_model="cross-encoder/nli-deberta-v3-base",
            embedding_model=EMB,
            top_k_chunks=None,  # keep all chunks so premise count == sentences
            top_k_premises=top_k_premises,
        )
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best * 1000.0, score


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sizes = [5, 10, 20, 40]
    k = 5

    print("Warming up models...")
    _time_score(_build_context(5), None, repeats=1)

    rows = []
    for n in sizes:
        ctx = _build_context(n)
        off_ms, off_score = _time_score(ctx, None)
        on_ms, on_score = _time_score(ctx, k)
        speedup = off_ms / on_ms if on_ms else float("nan")
        rows.append({
            "context_sentences": n,
            "off_ms": round(off_ms, 1),
            "on_ms": round(on_ms, 1),
            "speedup": round(speedup, 2),
            "off_score": round(off_score, 4),
            "on_score": round(on_score, 4),
            "score_delta": round(abs(off_score - on_score), 4),
        })
        print(f"  ctx={n:>2}  OFF {off_ms:8.1f}ms  ON(k={k}) {on_ms:8.1f}ms  "
              f"speedup {speedup:4.2f}x  score {off_score:.3f}->{on_score:.3f}")

    out = {"benchmark": "groundedness_latency_topk", "k": k, "rows": rows}
    p = RESULTS_DIR / "groundedness_latency.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
