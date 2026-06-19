#!/usr/bin/env python3
"""Workstream 2: diagnose why A0 responses score ~0.284 mean IQS."""
import json, sys, statistics
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Remove benchmarks/ from sys.path so it doesn't shadow the HuggingFace
# 'datasets' package that sentence-transformers imports internally.
_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from scroot import Auditor

PERTURBED = Path(__file__).parent / "datasets" / "nq_500_perturbed.jsonl"

# Load A0 samples
records = []
with PERTURBED.open(encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec["perturbation_level"] == 0:
            records.append(rec)
        if len(records) >= 20:
            break

auditor = Auditor()
results = []

for rec in records:
    result = auditor.score(
        query=rec["query"],
        response=rec["response"],
        context=[rec["context"]],
    )
    word_count = len(rec["response"].split())
    char_count = len(rec["response"])
    results.append({
        "query": rec["query"],
        "response": rec["response"],
        "response_words": word_count,
        "response_chars": char_count,
        "groundedness": result.groundedness,
        "completeness": result.completeness,
        "relevance": result.relevance,
        "consistency": result.consistency,
        "confidence": result.confidence,
        "iqs": result.iqs,
        "flags": result.flags,
    })
    print(f"Q: {rec['query'][:70]}...")
    print(f"R ({word_count}w): {rec['response'][:100]}...")
    print(f"  ground={result.groundedness:.3f}  complete={result.completeness:.3f}  "
          f"relevant={result.relevance:.3f}  consist={result.consistency:.3f}  "
          f"confid={result.confidence:.3f}  IQS={result.iqs:.3f}")
    print(f"  flags: {result.flags}")
    print()

# Statistics
metrics = ["groundedness", "completeness", "relevance", "consistency", "confidence", "iqs"]
print("\n=== MEANS ===")
for m in metrics:
    vals = [r[m] for r in results if r[m] is not None]
    print(f"  {m:<15} mean={statistics.mean(vals):.4f}  stdev={statistics.stdev(vals):.4f}  "
          f"min={min(vals):.4f}  max={max(vals):.4f}")

# Length correlation
words = [r["response_words"] for r in results]
iqs_vals = [r["iqs"] for r in results]
try:
    from scipy.stats import pearsonr
    r, p = pearsonr(words, iqs_vals)
    print(f"\n  Word count vs IQS: Pearson r={r:.4f} (p={p:.4f})")
except ImportError:
    pass

# Single vs multi sentence
single = [r["iqs"] for r in results if r["response"].count(".") <= 1]
multi  = [r["iqs"] for r in results if r["response"].count(".") > 1]
if single:
    print(f"\n  Single-sentence IQS: mean={statistics.mean(single):.4f} (n={len(single)})")
if multi:
    print(f"  Multi-sentence IQS:  mean={statistics.mean(multi):.4f} (n={len(multi)})")
