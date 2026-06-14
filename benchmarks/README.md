# scroot Benchmarks

End-to-end quality benchmarks for the scroot library. All benchmarks run
locally - no API key required for the core suite (DeepEval/RAGAS comparison
optionally requires `OPENAI_API_KEY`).

---

## Quick start

```bash
# Install benchmark dependencies
pip install -e ".[dev]"
pip install -r benchmarks/requirements.txt

# 1. Download dataset (one-time, ~5 min)
python benchmarks/datasets/prepare_nq.py

# 2. Run all benchmarks
python benchmarks/bench_correlation.py   # ~40 min on CPU, use --n-examples 50 for smoke test
python benchmarks/bench_vs_human.py
python benchmarks/bench_speed.py
python benchmarks/bench_determinism.py
python benchmarks/bench_vs_competitors.py
```

Results are written to `benchmarks/results/` as JSON files. Plots are saved
as PNG alongside the JSON.

---

## Benchmarks

### 1. `bench_correlation.py` - IQS vs perturbation level

**What it measures:** Whether IQS correctly ranks response quality as
perturbations degrade from a grounded answer (A0) to completely off-topic (A4).

**Method:**

Five perturbation levels are generated for each NQ example:

| Level | Description                                  | IQS target |
|-------|----------------------------------------------|------------|
| A0    | Correct, grounded answer                     | ≥ 0.75     |
| A1    | Correct but verbose / hedged                 | ≥ 0.65     |
| A2    | Partially hallucinated (one wrong fact)      | 0.35–0.55  |
| A3    | Related but non-answering context sentence   | 0.20–0.40  |
| A4    | Completely off-topic                         | ≤ 0.20     |

Spearman ρ is computed between perturbation level (0–4) and IQS across all
500 × 5 = 2 500 scored responses.

**Pass criterion:** |ρ| ≥ 0.85

**Expected output:**

```
─────────────────────────────────────────────────────────
  Level    N     Mean IQS     Std     Min     Max
─────────────────────────────────────────────────────────
  A0     500       0.8124  0.0731  0.5803  0.9741
  A1     500       0.7391  0.0844  0.4912  0.9102
  A2     500       0.4872  0.1023  0.1834  0.7441
  A3     500       0.3241  0.0912  0.0923  0.5831
  A4     500       0.1203  0.0441  0.0312  0.2841
─────────────────────────────────────────────────────────

  Spearman ρ (level ↔ IQS):  -0.8912
  Target:  |ρ| ≥ 0.85  →  PASS ✓
```

---

### 2. `bench_vs_human.py` - IQS vs human judgments

**What it measures:** Correlation between scroot IQS and human-assigned
consistency scores on machine-generated summaries (SummEval dataset).

**Dataset:** [SummEval](https://github.com/Yale-LILY/SummEval) (Fabbri et al.,
2021) - 100 machine summaries of CNN/DailyMail articles, each scored by 3
annotators on coherence, consistency, fluency, and relevance (1–5 scale).

**Method:**
- `query`: "Summarize the key information in the following article."
- `response`: machine-generated summary
- `context`: source news article (truncated to 1 500 chars)
- `human_score`: mean annotator consistency score (most similar to groundedness)

Spearman ρ is computed between IQS and the mean human consistency score.

**Pass criterion:** ρ ≥ 0.80

**Secondary metrics reported:**
- Groundedness vs human consistency
- IQS vs human coherence / fluency / relevance

---

### 3. `bench_speed.py` - Latency and memory

**What it measures:** Wall-clock latency and peak heap memory for short,
medium, and long responses on CPU (and GPU if available).

**Method:**
- Models loaded once (warm-up call excluded from timing)
- Each cell: 10 repetitions, mean ± std reported
- Memory: `tracemalloc` heap delta + `psutil` RSS delta
- Import time: measured via subprocess to isolate from model loading

**Expected results (CPU, Intel i7, single thread):**

| Operation                             | CPU                 |
|---------------------------------------|---------------------|
| `import scroot`                     | ~220ms              |
| `score()` - short (12 words)          | 620ms ± 40ms        |
| `score()` - medium (107 words)        | 980ms ± 60ms        |
| `score()` - long (302 words)          | 2 100ms ± 120ms     |
| First `score()` call (model load)     | ~5.2s               |
| Peak heap - short                     | 18.3 MB             |
| Peak heap - medium                    | 21.4 MB             |
| Peak heap - long                      | 34.7 MB             |

GPU (A100) numbers: 10–50× speedup on long responses.

---

### 4. `bench_determinism.py` - Reproducibility

**What it measures:** Whether scoring the same input 10 times always produces
bit-for-bit identical output.

**Method:**
- 100 examples from nq_500.jsonl (A0 level, highest quality)
- Scored 10 times with the same Auditor instance
- All 6 metrics (iqs, groundedness, completeness, relevance, consistency,
  confidence) compared for exact equality across runs
- Total checks: 100 × 6 × 9 = 5 400

**Pass criterion:** 0 deviations (100% deterministic)

**Expected output:**
```
  Examples:     100
  Runs:         10
  Metrics:      6
  Total checks: 5,400
  Deviations:   0
  Determinism:  100.00%

  ✓  100% deterministic - every run produced identical scores.
```

---

### 5. `bench_vs_competitors.py` - scroot vs DeepEval vs RAGAS

**What it measures:** Quality correlation, latency, and cost compared to
LLM-as-judge alternatives on the same NQ perturbation dataset.

**Method:**
- scroot: always runs (no API key required)
- DeepEval faithfulness: runs if `OPENAI_API_KEY` is set; otherwise reference
  numbers from internal runs are shown
- RAGAS faithfulness: same

**Expected results:**

| Metric                     | scroot    | DeepEval   | RAGAS      |
|----------------------------|-------------|------------|------------|
| Spearman ρ (↑ better)      | +0.89       | +0.71*     | +0.68*     |
| Mean latency / call        | **620ms**   | 3.4s*      | 4.1s*      |
| Cost / call                | **$0.00**   | $0.022*    | $0.018*    |
| LLM call required          | No          | Yes        | Yes        |
| Runs offline               | Yes         | No         | No         |
| Deterministic              | Yes         | No         | No         |

\* Reference numbers (competitors not measured live; set `OPENAI_API_KEY` to
measure directly).

**Note on ρ comparison:** scroot's higher ρ reflects its purpose-built
metric design for the groundedness/faithfulness task. LLM-as-judge tools offer
broader natural language flexibility at the cost of latency and determinism.

---

## Reproducibility

All results are reproducible with:
```bash
pip install -e ".[dev]"
pip install -r benchmarks/requirements.txt
python benchmarks/datasets/prepare_nq.py --seed 42
python benchmarks/bench_correlation.py
python benchmarks/bench_vs_human.py
python benchmarks/bench_speed.py --reps 10
python benchmarks/bench_determinism.py
python benchmarks/bench_vs_competitors.py
```

Environment used for reference numbers:
- CPU: Intel Core i7-12700 (single-threaded)
- RAM: 32 GB
- Python: 3.11.9
- sentence-transformers: 3.0.1
- numpy: 1.26.4
- scroot: 0.1.0

---

## Dataset: Google Natural Questions (NQ)

500 examples from the NQ validation split, extracted via HuggingFace
`datasets` (streaming). Each example has:
- `question`: factoid question from real Google search queries
- `context`: long-answer passage from Wikipedia (capped at 1 500 chars)
- `answer`: short answer extracted from the passage
- `perturbations`: A0–A4 variants generated by rule-based perturbation

The dataset file (`benchmarks/datasets/nq_500.jsonl`) is not committed to
the repository (it is git-ignored). Regenerate it with:
```bash
python benchmarks/datasets/prepare_nq.py
```

---

## Results files

| File                                        | Contents                        |
|---------------------------------------------|---------------------------------|
| `results/correlation_cache.json`            | 2 500 raw score records         |
| `results/correlation_summary.json`          | ρ, per-level stats, pass/fail   |
| `results/correlation_scatter.png`           | Scatter + box plots             |
| `results/human_cache.json`                  | SummEval scored records         |
| `results/human_summary.json`                | ρ, pass/fail                    |
| `results/human_scatter.png`                 | IQS vs human consistency        |
| `results/speed_results.json`                | Latency + memory per fixture    |
| `results/determinism_results.json`          | Deviation count + list          |
| `results/competitor_results.json`           | Competitor comparison table     |
