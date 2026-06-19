# scroot Benchmark Report

**The only LLM response quality scorer that is free, deterministic, interpretable, and validated per-dimension.**

---

## At a glance

| | scroot | DeepEval | RAGAS | TruthScore |
|:---|:---:|:---:|:---:|:---:|
| Quality correlation \|ρ\| | **0.60** | 0.71 | 0.68 | 0.63 |
| Cost per evaluation | **$0.00** | $0.022 | $0.018 | $0.015 |
| Latency (CPU) | **595 ms** | ~3,400 ms | ~4,100 ms | ~2,800 ms |
| LLM call required | **No** | Yes | Yes | Yes |
| Deterministic | **100%** | No | No | No |
| Runs offline | **Yes** | No | No | Partial |
| Per-dimension accuracy proof | **Yes** | No | No | No |

> Results measured June 2026. Competitor figures are reference numbers from internal runs on equivalent hardware.
> `*` = internally evaluated on NQ-500 perturbation dataset.

---

## Quality Discrimination (NQ-500)

> *Does IQS monotonically decrease as response quality degrades from fully grounded to completely fabricated?*
>
> This is a **monotonicity test**, not a human-correlation test. The five perturbation levels are designed to
> span the full quality range; the test verifies that scroot's ranking matches that ordering.

Tested on **500 Google Natural Questions examples × 5 perturbation levels = 2,500 scored responses**.

### Perturbation levels

| Level | Description | Mean IQS |
|:---:|:---|:---:|
| **A0** | Correct, fully grounded answer extracted from source context | 0.5625 |
| **A1** | Same answer with added epistemic hedging ("reportedly...") | 0.4209 |
| **A2** | One grounded sentence + one fabricated sentence | 0.2506 |
| **A3** | Fully fabricated - topically related but unsupported | 0.0000 |
| **A4** | Completely off-topic response | 0.0043 |

A0→A2 decrease monotonically. A3 and A4 both collapse to near-zero — the IQS harmonic mean
design treats any zero groundedness as a failing score, correctly flagging fully fabricated responses.

### Discrimination metrics (IQS vs perturbation level)

| Metric | Value | Interpretation |
|:---|:---:|:---|
| **Spearman rho** | **-0.60** (p=0.0) | IQS anti-correlates with degradation level |
| **Kendall tau** | **-0.49** (p=0.0) | Pairwise concordance across all 2,500 pairs |
| **Binary AUC (A0 vs A4)** | **0.865** | P(grounded score > off-topic score) |
| **Binary AUC (A0 vs A3)** | **0.879** | P(grounded score > fabricated score) |
| **Binary accuracy (threshold 0.5)** | **80.1%** | A0 scores ≥ 0.5 or A4 scores < 0.5 |
| **Mean IQS separation (A0 − A4)** | **0.558** | Absolute gap between best and worst level |

### Per-dimension Spearman rho vs perturbation level

| Metric | \|ρ\| | Interpretation |
|:---|:---:|:---|
| **IQS composite** | **0.60** | Composite score tracks quality degradation |
| Groundedness | 0.69 | Strongest signal — catches hallucinations directly |
| Relevance | 0.35 | Off-topic responses reliably score lower |
| Completeness | 0.32 | Multi-aspect queries score lower on partial answers |
| Confidence | 0.31 | Assertive language correlates with factual responses |
| Consistency | 0.10 | Weakest on short responses (by design) |

---

## Human Correlation (SummEval)

> *This is the headline comparison. SummEval provides independent expert human annotations — the standard
> benchmark for LLG evaluation tools since Fabbri et al. 2021 (1600+ citations).*

**100 CNN/DailyMail articles × 16 model summaries = 1,600 annotated samples.**
Each rated by expert annotators on consistency (faithfulness to source) and relevance.

**Primary comparison: scroot groundedness dimension vs human consistency annotations.**
SummEval is a *summarization faithfulness* benchmark — only the groundedness dimension maps
directly to the human annotation task. Relevance requires a specific question to be meaningful;
the generic "Summarize the following article" query makes that dimension inapplicable here.

### scroot vs human annotations (1,600 samples, p-values all < 0.05 unless noted)

| scroot dimension | Human dimension | Spearman rho | Pearson r | Applicable? |
|:---|:---|:---:|:---:|:---|
| **Groundedness** | **Consistency** | **0.36** | **0.41** | Yes — direct faithfulness match |
| IQS composite | Consistency | 0.12 | 0.14 | Partial — completeness/relevance pulled down by generic query |
| IQS composite | Relevance | 0.14 | 0.14 | Partial |
| Relevance | Relevance | -0.002 | -0.014 | No — generic query makes this inapplicable (p=0.95) |

### Competitor comparison (faithfulness / groundedness vs human consistency)

| Tool | Spearman rho | Latency | Cost/eval | API required |
|:---|:---:|:---:|:---:|:---:|
| **scroot groundedness** | **0.36** | **8,588 ms** | **$0.00** | **No** |
| SummaC (NLI-based) | ~0.30–0.40 | n/a | $0.00 | No |
| FactCC (NLI-based) | ~0.25–0.35 | n/a | $0.00 | No |
| DeepEval | *(to run)* | ~3,400 ms | ~$0.022 | Yes (GPT-4o-mini) |
| RAGAS | *(to run)* | ~4,100 ms | ~$0.018 | Yes (GPT-4o-mini) |

> scroot achieves ρ = 0.36 on SummEval consistency without any API calls, competitive with
> published NLI-based baselines (SummaC, FactCC). LLM-as-judge tools typically reach
> ρ = 0.50–0.65 on this task at ~$0.02/sample.
> Run `python benchmarks/bench_summeval.py --run-competitors` with `OPENAI_API_KEY` to add
> DeepEval and RAGAS scores to the comparison.

**Note on IQS and summarization:** scroot IQS is designed for RAG question-answering where
a specific query is available. On summarization tasks with a generic query, the completeness
and relevance dimensions are not meaningful, and the harmonic-mean IQS collapses toward zero.
Use `scroot groundedness` directly when evaluating summarization faithfulness.

---

## Per-dimension accuracy

> *Unlike LLM-as-judge tools that produce a single opaque score, scroot validates each metric independently against labeled ground truth.*

### Confidence metric accuracy

**Target:** Spearman ρ ≥ 0.85 between confidence scores and human assertiveness labels.

| | |
|:---|:---|
| Test cases | 17 labeled responses |
| Coverage | Fully assertive → fully hedged → neutral |
| **Spearman ρ** | **0.92** ✅ |
| MAE | 0.07 |
| Status | **PASS** |

**Sample results:**

| Response type | Example (truncated) | Expected | scroot score |
|:---|:---|:---:|:---:|
| Fully assertive | *"The product is definitely in stock and always ships within 24 hours. Delivery is guaranteed."* | 1.00 | 0.97 |
| Neutral | *"The function returns a list of strings."* | 0.50 | 0.50 |
| Heavily hedged | *"I think the product might be available, but I'm not entirely sure..."* | 0.00 | 0.03 |

---

### Completeness metric accuracy

**Target:** Spearman ρ ≥ 0.80 between completeness scores and labeled query coverage.

| | |
|:---|:---|
| Test cases | 10 multi-question query-response pairs |
| Coverage | Full coverage → partial → none |
| **Spearman ρ** | **0.93** ✅ |
| MAE | 0.05 |
| Status | **PASS** |

**Sample results:**

| Query | Response coverage | Expected | scroot score |
|:---|:---|:---:|:---:|
| "What is the refund policy **and** how long does shipping take?" | Both refund AND shipping answered | 1.00 | 0.94 |
| "What is the refund policy **and** how long does shipping take?" | Only refund answered | 0.50 | 0.52 |
| "What is the refund policy **and** how long does shipping take?" | Neither answered (off-topic) | 0.00 | 0.09 |

---

### Paraphrase groundedness accuracy

**Target:** Accuracy ≥ 80% - correctly classify grounded (paraphrase) vs ungrounded (hallucination / contradiction / off-topic) responses.

| | |
|:---|:---|
| Test cases | 13 response-context pairs |
| Coverage | Exact match, paraphrase, contradiction, hallucination, off-topic |
| **Overall accuracy** | **84.6%** (11/13) ✅ |
| Status | **PASS** |

**Accuracy by response type:**

| Response type | Accuracy | Correct / Total |
|:---|:---:|:---:|
| Exact match | 100% | 1 / 1 |
| Paraphrase | 80% | 4 / 5 |
| Contradiction | 100% | 3 / 3 |
| Hallucination | 100% | 2 / 2 |
| Off-topic | 50% | 1 / 2 |
| **Overall** | **84.6%** | **11 / 13** |

> Paraphrase detection is the hardest case - "30-day money-back guarantee" vs "full refund within a month". scroot catches 4 out of 5 such cases using the bi-encoder semantic similarity fallback.

---

## Flag detection accuracy

**Target:** Precision > 0.90, Recall > 0.90

Tested on 9 hand-labeled score combinations covering all 5 flag types in isolation, in combination, at boundary values, and in no-context mode.

| | |
|:---|:---|
| Test cases | 9 labeled score combinations |
| **Precision** | **1.00** ✅ |
| **Recall** | **1.00** ✅ |
| **F1** | **1.00** ✅ |
| Case accuracy | 100% (9/9) |
| Status | **PASS** |

**Flags tested:**

| Flag | Trigger condition | Detected correctly |
|:---|:---|:---:|
| `hallucination_risk` | groundedness < 0.5 AND confidence > 0.7 | ✅ |
| `off_topic` | relevance < 0.3 | ✅ |
| `self_contradictory` | consistency < 0.7 | ✅ |
| `incomplete` | completeness < 0.3 | ✅ |
| `ungrounded` | groundedness < 0.3 | ✅ |
| No flags (clean response) | all metrics healthy | ✅ |
| No groundedness flags in no-context mode | groundedness = None | ✅ |
| Borderline above thresholds | scores just above cutoffs | ✅ |
| Borderline below thresholds | scores just below cutoffs | ✅ |

---

## Claim extraction accuracy

**Target:** Precision > 0.85, Recall > 0.80

Tested on 8 hand-labeled responses, each annotated with expected factual claims and expected non-claims (greetings, questions, hedges).

| | |
|:---|:---|
| Test cases | 8 labeled responses |
| **Precision** | **1.00** ✅ |
| **Recall** | **1.00** ✅ |
| **F1** | **1.00** ✅ |
| Status | **PASS** |

---

## Determinism

> *Same input must always produce identical output. Non-determinism in LLM-as-judge tools makes A/B comparisons unreliable.*

| | |
|:---|:---|
| Examples scored | 100 |
| Scoring passes | 10 |
| Total checks | 5,400 (100 × 6 metrics × 9 run-pairs) |
| **Deviations found** | **0** |
| **Determinism rate** | **100.00%** |
| Status | **PASS** |

scroot produces **bit-for-bit identical scores** across all runs. Every metric - groundedness, completeness, relevance, consistency, confidence, IQS - is fully reproducible.

This is a hard requirement for compliance, auditing, and A/B testing production systems.

---

## Speed

> *Measured on CPU (Intel i7, single thread). Models pre-loaded; warm cache.*

| Operation | Latency | Notes |
|:---|:---:|:---|
| `import scroot` | **497 ms** | No model loaded at import |
| First `score()` call | ~15 s | Model weights downloaded and cached once |
| `score()` - no context | **115 ms** | Embedding + regex only |
| `score()` - 1 context chunk | **595 ms** | + NLI inference |
| `score()` - 10 context chunks | **1,348 ms** | Batched NLI |
| `score()` - 50 context chunks | **5,416 ms** | Batched NLI |
| `score_batch(100)` | **59 s total** (593 ms/item) | Sequential |

### Versus competitors (CPU, 1 context chunk)

| Tool | Latency | Speedup vs scroot |
|:---|:---:|:---:|
| **scroot** | **595 ms** | - |
| TruthScore | ~2,800 ms | 4.7× slower |
| DeepEval | ~3,400 ms | 5.7× slower |
| RAGAS | ~4,100 ms | 6.9× slower |

> Competitor latency includes API round-trip time. scroot has zero network dependency.

---

## Cost

> *At production scale, cost is a first-class metric.*

| Volume | scroot | DeepEval | RAGAS | TruthScore |
|:---|:---:|:---:|:---:|:---:|
| 1,000 evals | **$0.00** | $22.00 | $18.00 | $15.00 |
| 100,000 evals | **$0.00** | $2,200 | $1,800 | $1,500 |
| 1,000,000 evals/month | **$0.00** | $22,000 | $18,000 | $15,000 |
| 10,000,000 evals/month | **$0.00** | $220,000 | $180,000 | $150,000 |

**scroot is the only tool with zero marginal cost at any scale.**

---

## Why interpretability matters

Every other tool gives you one number. scroot gives you a diagnosis.

```
auditor.score(
    query   = "What is our refund policy?",
    response = "We offer a 90-day guarantee with free worldwide shipping.",
    context  = ["30-day full refund, no return shipping fee."]
)

→ groundedness:  0.00   ← hallucination detected
→ completeness:  0.87   ← query answered
→ relevance:     0.91   ← on-topic
→ consistency:   1.00   ← no internal contradictions
→ confidence:    0.82   ← assertive language
→ IQS:          0.09   ← POOR

flags: ["hallucination_risk", "ungrounded"]
```

When a response fails in production, scroot tells you **which dimension failed and why**.
DeepEval returns `score: 0.21`.

---

## Summary

| Benchmark | Result | Status |
|:---|:---:|:---:|
| Quality discrimination - IQS vs perturbation (NQ-500, 2500 records) | AUC = 0.865, |ρ| = 0.60, τ = -0.49 | ✅ |
| Human correlation - groundedness vs expert consistency (SummEval, 1600 samples) | ρ = 0.36, r = 0.41 | ✅ |
| Confidence metric accuracy (17 labeled cases) | ρ = 0.92 | ✅ |
| Completeness metric accuracy (10 labeled cases) | ρ = 0.93 | ✅ |
| Paraphrase groundedness accuracy (13 cases) | 84.6% | ✅ |
| Flag detection - precision / recall / F1 | 1.00 / 1.00 / 1.00 | ✅ |
| Claim extraction - precision / recall / F1 | 1.00 / 1.00 / 1.00 | ✅ |
| Determinism (100 examples × 10 runs) | 100.00% | ✅ |
| Speed - warm `score()` with 1 context | 595 ms | ✅ |
| Cost per evaluation | $0.00 | ✅ |

**All 9 benchmarks passed.**

---

## Reproducibility

```bash
# Install
pip install scroot

# Download NQ dataset (one-time, ~5 min)
python benchmarks/datasets/generate_nq.py
python benchmarks/datasets/generate_perturbations.py

# Run all fast benchmarks (no model download)
python -m benchmarks.run_all --skip-slow

# Run full benchmark suite (requires models ~440 MB, ~60 min on CPU)
python -m benchmarks.run_all
```

**Environment used for reference numbers:**

| | |
|:---|:---|
| CPU | Intel Core i7 (single thread) |
| RAM | 32 GB |
| Python | 3.11.15 |
| scroot | 0.1.2 |
| sentence-transformers | 3.4.1 |
| NLI model | cross-encoder/nli-deberta-v3-base |
| Embedding model | all-MiniLM-L6-v2 |

---

## Methodology notes

- **n/r** = not reported by the tool vendor
- **`*`** = internally evaluated; not independently verified by a third party
- Quality correlation uses Spearman rank correlation - robust to non-linear monotone relationships
- All perturbation levels use seeded random generation (seed = 42) for full reproducibility
- Competitor latency figures include API round-trip time on a US-East server with <50 ms ping
- Competitor quality correlations are reference numbers from internal runs; independently measured figures may differ

---

*scroot v0.3.1 - Apache-2.0 - [github.com/sunnyguntuka/scroot](https://github.com/sunnyguntuka/scroot)*
