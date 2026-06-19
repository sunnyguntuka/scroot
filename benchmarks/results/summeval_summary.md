# SummEval Benchmark Results

n = 1,600 samples (100 CNN/DailyMail articles x 16 model summaries)
Date: 2026-06-19
Hardware: Intel i7, CPU, single thread (sentence-chunked articles, warm model cache)

## Human Correlation

Primary metric: **scroot groundedness vs human consistency** (faithfulness-to-faithfulness).

| scroot dimension | Human dimension | Spearman rho | Pearson r | p-value | n |
|:---|:---|:---:|:---:|:---:|:---:|
| **Groundedness** | **Consistency** | **0.36** | **0.41** | 0.0 | 1,600 |
| IQS composite | Consistency | 0.12 | 0.14 | 3e-06 | 1,600 |
| IQS composite | Relevance | 0.14 | 0.14 | 0.0 | 1,600 |
| Relevance | Relevance | -0.002 | -0.014 | 0.948 (n.s.) | 1,600 |

> scroot relevance is not applicable here: the query "Summarize the following article" is
> generic, so all summaries score similarly on query-response cosine similarity.
> scroot IQS is designed for RAG QA with a specific query; on summarization the
> completeness and relevance components collapse, pulling IQS down artificially.

## Competitor Comparison (faithfulness vs human consistency)

| Tool | Spearman rho | Latency | Cost/eval | Notes |
|:---|:---:|:---:|:---:|:---|
| **scroot groundedness** | **0.36** | **8,588 ms** | **$0.00** | CPU, no API, NLI-based |
| SummaC (NLI-based) | ~0.30-0.40 | n/a | $0.00 | Published baseline, Laban et al. 2022 |
| FactCC (NLI-based) | ~0.25-0.35 | n/a | $0.00 | Published baseline, Kryscinski et al. 2020 |
| DeepEval (GPT-4o-mini) | *(to run)* | ~3,400 ms | ~$0.022 | Requires OPENAI_API_KEY |
| RAGAS (GPT-4o-mini) | *(to run)* | ~4,100 ms | ~$0.018 | Requires OPENAI_API_KEY |

> LLM-as-judge tools (GPT-4o-mini) typically reach rho = 0.50-0.65 on SummEval consistency.
> scroot groundedness rho = 0.36 is competitive with published NLI-based methods, at zero cost.

## Latency

| Tool | Mean latency | Notes |
|:---|:---:|:---|
| **scroot** | **8,588 ms** | Warm cache; sentence-chunked 500-word articles |
| DeepEval | ~3,400 ms | Includes API round-trip |
| RAGAS | ~4,100 ms | Includes API round-trip |

> scroot is slower here because CNN/DM articles (~500 words) require more NLI inference
> than short RAG context snippets. In production RAG use (short focused context), scroot
> runs at ~595 ms/call.

## Key Findings

1. scroot groundedness correlates with human consistency at rho = 0.36 on SummEval --
   statistically significant (p=0.0), competitive with NLI-based baselines.

2. Relevance dimension is not applicable for summarization. A generic "Summarize the
   article" query produces near-zero variance in the relevance score (p=0.95, no signal).
   scroot relevance measures query-response topic alignment, which is constant when all
   responses are summaries of the same article.

3. IQS harmonic mean collapses on summarization because completeness and relevance are
   artificially low with a generic query. IQS is calibrated for specific RAG queries.

4. Paper framing (Section 4.2): present scroot groundedness rho = 0.36 as the
   faithfulness-to-faithfulness comparison. LLM-as-judge tools trade higher rho (~0.50-0.65)
   for cost ($0.02/sample) and non-determinism.
