# Benchmark methodology

This page documents how the benchmark numbers quoted in the
[README](../README.md) and [`benchmarks/README.md`](../benchmarks/README.md)
were produced, so the claims can be independently reproduced or
challenged.

## Hardware and environment

Reference numbers (latency, memory, "vs DeepEval/RAGAS" comparisons) were
measured on:

- CPU: Intel Core i7-12700 (single-threaded)
- RAM: 32 GB
- OS: Windows 11 / Ubuntu 22.04 (CI)
- Python: 3.11.9
- scroot: 0.1.0
- sentence-transformers: 3.0.1
- numpy: 1.26.4

GPU figures (where quoted) were measured on an NVIDIA A100.

## Dataset

The core suite uses 500 examples from the [Google Natural Questions
(NQ)](https://ai.google.com/research/NaturalQuestions) validation split,
streamed via the HuggingFace `datasets` library. Each example is expanded
into five perturbation levels (A0-A4, from a fully grounded answer to a
completely off-topic one), giving 2,500 scored responses for the
correlation benchmark.

The human-correlation benchmark uses
[SummEval](https://github.com/Yale-LILY/SummEval) (Fabbri et al., 2021):
100 machine-generated summaries of CNN/DailyMail articles, each rated by 3
human annotators on a 1-5 scale.

Dataset preparation is deterministic given a fixed seed:

```bash
python benchmarks/datasets/prepare_nq.py --seed 42
```

## How IQS is computed

The Information Quality Score (IQS) is the **weighted harmonic mean** of
five sub-metrics, implemented in
[`src/scroot/composite.py`](../src/scroot/composite.py):

```
IQS = n / sum(w_i / s_i),  where n = sum(w_i)
```

Default weights:

| Metric        | Weight |
|---------------|--------|
| Groundedness  | 0.35   |
| Completeness  | 0.25   |
| Relevance     | 0.20   |
| Consistency   | 0.15   |
| Confidence    | 0.05   |

The harmonic mean is used (rather than an arithmetic mean) because it is
"zero-tolerant": a single sub-metric near zero (e.g. a hallucinated claim
driving groundedness toward 0) pulls IQS toward zero even if the other
metrics are high. When no `context` is supplied, the groundedness weight
is redistributed proportionally across the remaining four metrics. A
`mode="geometric"` option computes a weighted geometric mean instead, for
callers who want partial-quality responses to score more gently.

## How competitor numbers were estimated

`benchmarks/bench_vs_competitors.py` scores the same 50-example,
5-perturbation-level subset (250 responses) with scroot and, when
`OPENAI_API_KEY` is set, with DeepEval and RAGAS using `gpt-4o-mini` as the
judge model. Cost per evaluation is taken directly from the OpenAI billing
for that run (input + output tokens at `gpt-4o-mini` pricing at the time of
the run).

When `OPENAI_API_KEY` is not set, the published reference numbers below are
shown instead, each tagged with its source run:

| Tool       | Spearman ρ | Mean latency | Cost / call | Source |
|------------|-----------|--------------|-------------|--------|
| scroot   | +0.89     | 620ms        | $0.00       | live (no LLM call) |
| DeepEval   | +0.71     | 3.4s         | $0.022      | DeepEval v1.x, GPT-4o-mini, NQ-500 internal run |
| RAGAS      | +0.68     | 4.1s         | $0.018      | RAGAS v0.1.x, GPT-4o-mini, NQ-500 internal run |
| TruthScore | +0.63     | 2.8s         | $0.015      | TruthScore v0.2, GPT-4o-mini, NQ-500 internal run |

These reference numbers are from internal runs and are **not** re-measured
on every CI run; treat them as illustrative until you reproduce them with
your own `OPENAI_API_KEY`.

## How to reproduce

```bash
pip install -e ".[dev]"
pip install -r benchmarks/requirements.txt

python benchmarks/datasets/prepare_nq.py --seed 42
python benchmarks/bench_correlation.py
python benchmarks/bench_vs_human.py
python benchmarks/bench_speed.py --reps 10
python benchmarks/bench_determinism.py
python benchmarks/bench_vs_competitors.py   # add OPENAI_API_KEY for live competitor numbers
```

Results are written as JSON (and PNG plots) to `benchmarks/results/`. See
[`benchmarks/README.md`](../benchmarks/README.md) for the full
per-benchmark methodology, pass criteria, and expected output.

## Last run

The reference numbers in this document and in the README were last
measured for **scroot v0.1.0** (2026). If you notice numbers that look
stale relative to the current release, please open an issue - re-running
`benchmarks/run_all.py` and updating this file is part of the release
checklist.
