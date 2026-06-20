"""
Task 2 - RAGAS faithfulness on the SAME 396 matched samples as Task 1.

The yesterday's sprint could not run RAGAS: ragas 0.4.3 imports
`langchain_community.chat_models.vertexai`, a path removed in
langchain-community 0.4.x (the version in the main env, alongside langchain 1.x).
ragas>=0.5 does not exist on PyPI (0.4.3 is the latest release), so an
in-place upgrade is impossible.

Fix: run RAGAS from an isolated venv (.ragas-env) pinned to
  ragas==0.4.3, langchain<1.0 (resolved 0.2.17),
  langchain-community<0.3 (resolved 0.2.19), openai>=1.0
which restores the vertexai import path. THIS SCRIPT MUST BE RUN WITH THAT
VENV'S PYTHON: .ragas-env/Scripts/python.exe benchmarks/bench_ragas_matched.py

It scores RAGAS faithfulness (gpt-4o-mini judge) on the exact 396
(doc_id, summary_idx) pairs DeepEval scored (read from summeval_competitors.json),
correlates vs human consistency, and writes:
  benchmarks/results/ragas_matched.json
  benchmarks/results/ragas_matched.md

Cost guard: aborts before scoring if projected cost > $30.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# benchmarks/ shadows the HuggingFace `datasets` package -- drop it from path.
_bench_dir = str(Path(__file__).parent)
if _bench_dir in sys.path:
    sys.path.remove(_bench_dir)

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "datasets" / "summeval.jsonl"
COMPETITORS_PATH = RESULTS_DIR / "summeval_competitors.json"
OUT_JSON = RESULTS_DIR / "ragas_matched.json"
OUT_MD = RESULTS_DIR / "ragas_matched.md"

QUERY = "Summarize the following article."
COST_GUARD_USD = 30.0
# gpt-4o-mini pricing (USD per 1M tokens), 2025/2026.
PRICE_IN = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000


def _get_api_key() -> str:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        val, _ = winreg.QueryValueEx(key, "OPENAI_API_KEY")
        winreg.CloseKey(key)
        return val
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")


os.environ["OPENAI_API_KEY"] = _get_api_key()


def _chunk_article(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = [s.strip() for s in sents if len(s.split()) >= 5]
    return chunks if chunks else [text]


def _load_summeval() -> dict:
    by_key = {}
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["summary_idx"] = int(d["summary_idx"])
            d["human_consistency"] = float(d["human_consistency"])
            by_key[(d["doc_id"], d["summary_idx"])] = d
    return by_key


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not available from Windows user env.")
        sys.exit(2)

    comp = json.load(open(COMPETITORS_PATH, encoding="utf-8"))
    matched_keys = [
        (d["doc_id"], int(d["summary_idx"]))
        for d in comp["deepeval_raw"]["per_sample"]
    ]
    print(f"DeepEval matched samples to score with RAGAS: {len(matched_keys)}")

    by_key = _load_summeval()
    records = []
    for k in matched_keys:
        d = by_key.get(k)
        if d is None:
            print(f"  WARN missing in dataset: {k}")
            continue
        records.append(d)
    print(f"Records resolved: {len(records)}")

    import ragas
    print(f"RAGAS version: {ragas.__version__}", flush=True)
    from ragas import evaluate
    from ragas.metrics import faithfulness
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from langchain_openai import ChatOpenAI
    from langchain_community.callbacks.manager import get_openai_callback

    samples = [
        SingleTurnSample(
            user_input=QUERY,
            response=r["summary"],
            retrieved_contexts=_chunk_article(r["source"]),
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    out = {
        "tool": "RAGAS",
        "metric": "faithfulness",
        "judge": "gpt-4o-mini",
        "ragas_version": ragas.__version__,
        "venv": ".ragas-env (ragas==0.4.3, langchain 0.2.17, "
                "langchain-community 0.2.19)",
        "n_requested": len(records),
        "per_sample": [],
        "errors": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "note": "",
    }

    print("Scoring RAGAS faithfulness (batch evaluate)...", flush=True)
    ts = time.perf_counter()
    try:
        with get_openai_callback() as cb:
            result = evaluate(dataset, metrics=[faithfulness], llm=llm)
        elapsed = time.perf_counter() - ts
        out["tokens_in"] = cb.prompt_tokens
        out["tokens_out"] = cb.completion_tokens
        # Prefer callback's own cost; fall back to manual price calc.
        cost = cb.total_cost or (
            cb.prompt_tokens * PRICE_IN + cb.completion_tokens * PRICE_OUT
        )
        out["cost_usd"] = round(cost, 4)
        if cost > COST_GUARD_USD:
            print(f"COST GUARD: ${cost:.2f} > ${COST_GUARD_USD}")
    except Exception as e:  # noqa: BLE001
        out["note"] = f"RAGAS evaluate failed: {type(e).__name__}: {str(e)[:300]}"
        json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
        print(out["note"])
        sys.exit(1)

    df = result.to_pandas()
    col = "faithfulness"
    for r, val in zip(records, df[col].tolist()):
        if val is None or (isinstance(val, float) and val != val):  # NaN
            out["errors"].append(
                {"doc_id": r["doc_id"], "summary_idx": r["summary_idx"],
                 "error": "NaN score"}
            )
            continue
        out["per_sample"].append({
            "doc_id": r["doc_id"],
            "summary_idx": r["summary_idx"],
            "score": float(val),
            "latency_ms": round(elapsed / len(records) * 1000.0, 1),
            "human_consistency": r["human_consistency"],
        })
    out["mean_latency_ms"] = round(elapsed / len(records) * 1000.0, 1)
    out["note"] = (f"batch evaluate in {elapsed/60:.1f} min; "
                   f"{len(out['per_sample'])} scored, "
                   f"{len(out['errors'])} NaN/excluded")

    # Correlation vs human consistency.
    from scipy import stats
    scores = [p["score"] for p in out["per_sample"]]
    humans = [p["human_consistency"] for p in out["per_sample"]]
    if len(scores) >= 3:
        rho, rho_p = stats.spearmanr(scores, humans)
        r, r_p = stats.pearsonr(scores, humans)
        out["spearman"] = {"rho": round(float(rho), 4), "p": float(f"{rho_p:.3g}")}
        out["pearson"] = {"r": round(float(r), 4), "p": float(f"{r_p:.3g}")}
    else:
        out["spearman"] = None
        out["pearson"] = None

    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=2)

    def fmt_p(p):
        return "<0.001" if p is not None and p < 0.001 else f"{p:.3g}"

    lines = []
    lines.append("# RAGAS faithfulness on 396 matched samples (Task 2)")
    lines.append("")
    lines.append(
        "RAGAS `faithfulness` (gpt-4o-mini judge) on the SAME "
        f"{out['n_requested']} (doc_id, summary_idx) pairs DeepEval scored, "
        "correlated against the human `consistency` annotation."
    )
    lines.append("")
    lines.append("## Environment fix")
    lines.append("")
    lines.append(
        "ragas 0.4.3 (latest on PyPI; no >=0.5 exists) imports "
        "`langchain_community.chat_models.vertexai`, removed in "
        "langchain-community 0.4.x present in the main env. Resolved by running "
        "RAGAS from an isolated venv (`.ragas-env`) pinned to "
        "ragas==0.4.3 + langchain 0.2.17 + langchain-community 0.2.19 + "
        "openai 1.109, which restores the import path."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- Scored: **{len(out['per_sample'])}** / {out['n_requested']}")
    lines.append(f"- Excluded (NaN): {len(out['errors'])}")
    lines.append(f"- Tokens in/out: {out['tokens_in']:,} / {out['tokens_out']:,}")
    lines.append(f"- Cost: ${out['cost_usd']:.4f}")
    lines.append(f"- Mean latency/sample: {out.get('mean_latency_ms')} ms")
    lines.append("")
    if out.get("spearman"):
        lines.append("| Tool | Spearman rho | p | Pearson r | p |")
        lines.append("|------|-------------|---|-----------|---|")
        lines.append(
            f"| RAGAS faithfulness (gpt-4o-mini) | "
            f"{out['spearman']['rho']:.4f} | {fmt_p(out['spearman']['p'])} | "
            f"{out['pearson']['r']:.4f} | {fmt_p(out['pearson']['p'])} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== RAGAS RESULT ===")
    print(f"n_scored={len(out['per_sample'])} excluded={len(out['errors'])}")
    if out.get("spearman"):
        print(f"spearman rho={out['spearman']['rho']:.4f} p={out['spearman']['p']}")
        print(f"pearson r={out['pearson']['r']:.4f}")
    print(f"cost=${out['cost_usd']:.4f}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
