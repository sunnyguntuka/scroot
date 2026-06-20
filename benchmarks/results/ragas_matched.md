# RAGAS faithfulness on 396 matched samples (Task 2)

RAGAS `faithfulness` (gpt-4o-mini judge) on the SAME 396 (doc_id, summary_idx) pairs DeepEval scored, correlated against the human `consistency` annotation.

## Environment fix

ragas 0.4.3 (latest on PyPI; no >=0.5 exists) imports `langchain_community.chat_models.vertexai`, removed in langchain-community 0.4.x present in the main env. Resolved by running RAGAS from an isolated venv (`.ragas-env`) pinned to ragas==0.4.3 + langchain 0.2.17 + langchain-community 0.2.19 + openai 1.109, which restores the import path.

## Result

- Scored: **396** / 396
- Excluded (NaN): 0
- Tokens in/out: 696,017 / 167,811
- Cost: $0.2051
- Mean latency/sample: 389.6 ms

| Tool | Spearman rho | p | Pearson r | p |
|------|-------------|---|-----------|---|
| RAGAS faithfulness (gpt-4o-mini) | 0.6440 | <0.001 | 0.7301 | <0.001 |
