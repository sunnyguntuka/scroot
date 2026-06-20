# TruthScore: install attempt and formal exclusion (Task 3)

## What was investigated

The sprint brief listed **TruthScore** alongside DeepEval and RAGAS as an
"LLM-free faithfulness scorer," and the original sprint assumed
`pip install truthscore`. We researched whether that package exists and what
it actually is, plus the common alternates the name could have referred to.

| Candidate | On PyPI? | What it is |
|-----------|----------|------------|
| `truthscore` | Yes (0.3.0) | "Fast, modular reimplementation of RAGAS's FactualCorrectness metric." Decomposes the response into atomic claims **with an LLM**, then scores entailment with an NLI model. Depends on `ragas>=0.2.15,<1.0` + `sentence-transformers`. Author: Giovanni Gatti Pinheiro. MIT. |
| `mini-check` / `minicheck` | No | Not published under either name on PyPI. (MiniCheck is a research repo, not a pip package.) |
| `alignscore` | No | No PyPI distribution (research repo only). |
| `trulens` / `trulens-eval` | Yes | An observability/eval framework, not a single faithfulness metric; its groundedness feedback uses a hosted LLM judge. Not the described tool. |

## Install attempt

`truthscore==0.3.0` **is** installable. A dependency-resolution dry run inside
the isolated `.ragas-env` (which already pins a compatible `ragas==0.4.3`)
resolves cleanly:

```
.ragas-env/Scripts/python.exe -m pip install truthscore --dry-run
-> would install truthscore-0.3.0 plus torch-2.12.1, transformers-4.57.6,
   sentence-transformers-4.1.0, scikit-learn, tokenizers, safetensors, sympy
```

So the install is feasible (no version conflict). The exclusion below is **not**
an "it won't install" exclusion -- it is a *role/validity* exclusion.

## Why it is formally excluded from the comparison table

1. **It is not LLM-free.** The sprint brief's premise -- that TruthScore is an
   LLM-free faithfulness scorer comparable to scroot -- is incorrect for this
   package. `truthscore` requires an LLM for the claim-decomposition step
   (it reimplements RAGAS FactualCorrectness, which is LLM-driven). Listing it
   as an LLM-free baseline would misrepresent it.

2. **It is redundant with the RAGAS result we already report.** `truthscore`
   is, by its own description, a reimplementation of a RAGAS metric and depends
   on `ragas` itself, using the same claim-extraction LLM. Its column would
   measure essentially the same construct as the RAGAS `faithfulness` figure
   already in the table (Task 2, rho = 0.64), via the same judge family. It
   would add an apparent third data point that is not statistically
   independent of the RAGAS one -- misleading in a paper-grade comparison.

3. **The only LLM-free, independent faithfulness baseline in scope is scroot
   itself.** Among the named candidates, none is both (a) a pip-installable
   package and (b) genuinely LLM-free and independent. `alignscore` and
   `minicheck` -- the two that *are* LLM-free NLI-based faithfulness scorers --
   have no PyPI distribution and would require vendoring research code and
   downloading model checkpoints, which is out of scope for this sprint and was
   not the package the brief pointed at.

## Formal exclusion statement (paper)

> TruthScore was excluded from the head-to-head comparison. The pip package of
> that name (`truthscore` 0.3.0) is a reimplementation of RAGAS's
> FactualCorrectness metric: it decomposes responses into atomic claims using a
> large language model before NLI entailment scoring, and depends on the `ragas`
> library. It is therefore neither LLM-free (contrary to the role it was
> nominated for) nor statistically independent of the RAGAS faithfulness baseline
> already reported, so including it would double-count the RAGAS construct. The
> genuinely LLM-free NLI faithfulness scorers it could be confused with
> (AlignScore, MiniCheck) are not distributed on PyPI and were out of scope.
> scroot remains the only LLM-free, judge-independent faithfulness scorer in the
> comparison.

## Reproduction

- `pip index versions truthscore` -> 0.3.0 (latest), 0.2.0, 0.1.1, 0.1.0
- `pip install truthscore --dry-run` succeeds (no conflict with ragas 0.4.3)
- PyPI metadata: https://pypi.org/pypi/truthscore/json
  ("reimplementation of RAGAS's FactualCorrectness", deps include
  `ragas>=0.2.15,<1.0.0`, `sentence-transformers>=4.1.0,<5.0.0`)
