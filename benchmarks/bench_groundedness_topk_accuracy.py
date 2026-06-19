"""
Task 3 - accuracy + determinism validation for top-k premise pre-filtering.

Gate (must all pass):
  - Mean absolute groundedness-score difference (top-k ON vs OFF) < 0.02 on a
    random 50-sample NQ subset.
  - Determinism: 10 passes over 10 samples produce 0 deviations.

NQ-500 per-sample scores live in correlation_samples.jsonl but without the raw
text, so we re-score from the nq_500.jsonl dataset (query/context/answer) which
is joinable by id. We use the reference_answer as the "response" to ground
against its own context - a stable, deterministic re-scoring target.

Run:
  set PYTHONIOENCODING=utf-8
  python benchmarks/bench_groundedness_topk_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from scroot.metrics.groundedness import score_groundedness  # noqa: E402

DATASETS = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"
NQ = DATASETS / "nq_500.jsonl"
EMB = "all-MiniLM-L6-v2"


def _chunk(text: str) -> list[str]:
    import re
    s = re.split(r"(?<=[.!?])\s+", text.strip())
    return [x.strip() for x in s if len(x.split()) >= 4] or [text]


def _score(resp, ctx, k):
    s, _ = score_groundedness(
        resp, ctx,
        nli_model="cross-encoder/nli-deberta-v3-base",
        embedding_model=EMB,
        top_k_chunks=None,
        top_k_premises=k,
    )
    return s


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in NQ.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Deterministic "random" subset: every 10th record -> 50 samples.
    subset = rows[::10][:50]
    samples = [(r["reference_answer"], _chunk(r["context"])) for r in subset]

    # --- Accuracy: MAD between OFF and each k in {3,5,8,10} ---
    print("Accuracy: top-k ON vs OFF (50 samples)")
    mad_by_k = {}
    for k in (3, 5, 8, 10):
        diffs = []
        for resp, ctx in samples:
            off = _score(resp, ctx, None)
            on = _score(resp, ctx, k)
            diffs.append(abs(off - on))
        mad = sum(diffs) / len(diffs)
        maxd = max(diffs)
        mad_by_k[k] = {"mad": round(mad, 5), "max_abs_diff": round(maxd, 5)}
        print(f"  k={k:>2}  MAD={mad:.5f}  max={maxd:.5f}  "
              f"{'PASS' if mad < 0.02 else 'FAIL'}")

    # --- Determinism: 10 passes over first 10 samples, k=5 ---
    print("\nDeterminism: 10 passes x 10 samples (k=5)")
    det_samples = samples[:10]
    baseline = [_score(r, c, 5) for r, c in det_samples]
    deviations = 0
    for p in range(10):
        run = [_score(r, c, 5) for r, c in det_samples]
        for a, b in zip(baseline, run):
            if a != b:
                deviations += 1
    print(f"  deviations: {deviations} (0 required)")

    mad5 = mad_by_k[5]["mad"]
    acc_gate = all(v["mad"] < 0.02 for v in mad_by_k.values())
    det_gate = deviations == 0

    payload = {
        "benchmark": "groundedness_topk_accuracy",
        "n_samples": len(samples),
        "mad_by_k": mad_by_k,
        "determinism_deviations": deviations,
        "gates": {"mad_lt_0.02": acc_gate, "determinism_zero": det_gate},
    }
    p = RESULTS_DIR / "groundedness_topk_accuracy.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== GATES ===")
    print(f"  MAD < 0.02 (all k): {'PASS' if acc_gate else 'FAIL'} "
          f"(k=5 MAD={mad5:.5f})")
    print(f"  Determinism 0 deviations: {'PASS' if det_gate else 'FAIL'}")
    print(f"Saved -> {p}")
    sys.exit(0 if (acc_gate and det_gate) else 1)


if __name__ == "__main__":
    main()
