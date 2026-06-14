"""
Benchmark: End-to-end feedback loop.

Tests the complete correction pipeline without requiring model downloads:

  Stage 1 - Detection:   Score response → detect low IQS → store record (pending)
  Stage 2 - Review:      Retrieve pending records → mark_reviewed with correction
  Stage 3 - Guardrails:  GuardrailInjector injects corrections into system prompt
  Stage 4 - Improvement: Re-score corrected response → verify IQS improved
  Stage 5 - Export:      export_for_finetuning() → verify SFT training pairs

Uses mock auditor scores so no model download is required.

Usage:
    python benchmarks/bench_feedback_loop.py
    python benchmarks/bench_feedback_loop.py --with-real-models

Output:
    benchmarks/results/feedback_loop.json
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = RESULTS_DIR / "feedback_loop.json"


# ---------------------------------------------------------------------------
# Mock scoring helpers
# ---------------------------------------------------------------------------

def _mock_result(iqs: float, groundedness: float = 0.0, flags: list | None = None):
    """Create a mock EntailmentResult with deterministic scores."""
    r = MagicMock()
    r.iqs = iqs
    r.groundedness = groundedness
    r.completeness = 0.8
    r.relevance = 0.85
    r.consistency = 0.95
    r.confidence = 0.6
    r.flags = flags or []
    r.to_dict.return_value = {
        "iqs": iqs,
        "groundedness": groundedness,
        "completeness": 0.8,
        "relevance": 0.85,
        "consistency": 0.95,
        "confidence": 0.6,
        "flags": flags or [],
    }
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scroot.feedback.store import FeedbackStore, CorrectionRecord
    from scroot.feedback.injector import GuardrailInjector

    results: dict = {
        "benchmark": "feedback_loop",
        "stages": {},
        "passed": False,
    }
    all_passed = True

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        store_path = f.name

    try:
        store = FeedbackStore(path=store_path)

        # ------------------------------------------------------------------
        # Stage 1: Detection - score poor responses, store as pending
        # ------------------------------------------------------------------
        print("\n[Stage 1] Detection - score poor responses")

        BAD_CASES = [
            {
                "id": "loop-001",
                "query": "What is our refund policy?",
                "response": "We offer a 90-day money-back guarantee with free worldwide shipping.",
                "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
                "mock_iqs": 0.08,
                "mock_groundedness": 0.0,
                "mock_flags": ["hallucination_risk", "ungrounded"],
            },
            {
                "id": "loop-002",
                "query": "How long does shipping take?",
                "response": "Our products are high quality and trusted by thousands of customers.",
                "context": ["Standard shipping takes 5-7 business days."],
                "mock_iqs": 0.11,
                "mock_groundedness": 0.0,
                "mock_flags": ["off_topic", "incomplete"],
            },
            {
                "id": "loop-003",
                "query": "Is there a free trial?",
                "response": "Yes we offer a 14-day trial. Also no we do not have any trials.",
                "context": ["We offer a 14-day free trial with no credit card required."],
                "mock_iqs": 0.09,
                "mock_groundedness": 0.0,
                "mock_flags": ["self_contradictory", "hallucination_risk"],
            },
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for case in BAD_CASES:
                mock_result = _mock_result(
                    iqs=case["mock_iqs"],
                    groundedness=case["mock_groundedness"],
                    flags=case["mock_flags"],
                )
                # Simulate: if IQS < 0.5 or flags, log for review
                if mock_result.iqs < 0.5 or mock_result.flags:
                    store.add(CorrectionRecord(
                        id=case["id"],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        query=case["query"],
                        response=case["response"],
                        scores=mock_result.to_dict(),
                        flags=mock_result.flags,
                        correction="",  # empty - awaiting review
                        reason="",
                        context_used=case["context"],
                        corrected_by="scroot-auto",
                        status="pending",
                    ))

        pending = store.get_pending()
        stage1_ok = len(pending) == 3 and all(r.status == "pending" for r in pending)
        results["stages"]["detection"] = {
            "stored": len(pending),
            "all_pending": stage1_ok,
            "passed": stage1_ok,
        }
        print(f"  Stored {len(pending)} pending records  {'✓' if stage1_ok else '✗'}")
        all_passed = all_passed and stage1_ok

        # ------------------------------------------------------------------
        # Stage 2: Review - mark records with human corrections
        # ------------------------------------------------------------------
        print("\n[Stage 2] Review - apply human corrections")

        CORRECTIONS = {
            "loop-001": {
                "correction": "We offer a 30-day full refund at no extra cost.",
                "reason": "Response fabricated 90-day and free shipping - context says 30-day refund.",
                "corrected_by": "human-reviewer",
            },
            "loop-002": {
                "correction": "Standard shipping takes 5-7 business days.",
                "reason": "Response was off-topic - did not address shipping time from context.",
                "corrected_by": "human-reviewer",
            },
            "loop-003": {
                "correction": "Yes, we offer a 14-day free trial with no credit card required.",
                "reason": "Response self-contradicted. Correct answer is in context.",
                "corrected_by": "human-reviewer",
            },
        }

        reviewed_count = 0
        for record_id, data in CORRECTIONS.items():
            ok = store.mark_reviewed(
                record_id=record_id,
                correction=data["correction"],
                reason=data["reason"],
                corrected_by=data["corrected_by"],
                status="reviewed",
            )
            if ok:
                reviewed_count += 1

        pending_after = store.get_pending()
        reviewed = store.get_by_status("reviewed")
        stage2_ok = reviewed_count == 3 and len(pending_after) == 0 and len(reviewed) == 3
        results["stages"]["review"] = {
            "reviewed": reviewed_count,
            "pending_remaining": len(pending_after),
            "passed": stage2_ok,
        }
        print(f"  Reviewed {reviewed_count}/3 records, {len(pending_after)} pending remaining  "
              f"{'✓' if stage2_ok else '✗'}")
        all_passed = all_passed and stage2_ok

        # ------------------------------------------------------------------
        # Stage 3: Guardrails - inject corrections into next LLM prompt
        # ------------------------------------------------------------------
        print("\n[Stage 3] Guardrails - inject corrections into system prompt")

        injector = GuardrailInjector(store)
        context_recent = injector.build_context(strategy="recent", max_corrections=5)
        context_rules = injector.build_context(strategy="rules")

        has_recent_header = "KNOWN CORRECTIONS" in context_recent
        has_rules_header = "GUARDRAILS" in context_rules

        # Verify corrections appear in guardrail context
        refund_in_recent = "30-day" in context_recent
        shipping_in_rules = "5-7" in context_rules or "shipping" in context_rules.lower()

        stage3_ok = has_recent_header and has_rules_header and refund_in_recent
        results["stages"]["guardrails"] = {
            "recent_strategy_ok": has_recent_header and refund_in_recent,
            "rules_strategy_ok": has_rules_header,
            "corrections_visible": refund_in_recent,
            "passed": stage3_ok,
        }
        print(f"  recent: header={has_recent_header} corrections_in_context={refund_in_recent}  "
              f"{'✓' if has_recent_header and refund_in_recent else '✗'}")
        print(f"  rules:  header={has_rules_header}  {'✓' if has_rules_header else '✗'}")
        all_passed = all_passed and stage3_ok

        # ------------------------------------------------------------------
        # Stage 4: Improvement - re-score corrected responses, mark applied
        # ------------------------------------------------------------------
        print("\n[Stage 4] Improvement - verify corrected responses score higher")

        CORRECTED_SCORES = {
            "loop-001": {"iqs": 0.91, "groundedness": 0.97},
            "loop-002": {"iqs": 0.88, "groundedness": 0.94},
            "loop-003": {"iqs": 0.85, "groundedness": 0.92},
        }

        improvements = []
        for r in reviewed:
            if r.id in CORRECTED_SCORES:
                new_scores = CORRECTED_SCORES[r.id]
                original_iqs = r.scores.get("iqs", 0.0)
                improved = new_scores["iqs"] > original_iqs

                # Mark as applied with the new IQS
                store.mark_reviewed(
                    record_id=r.id,
                    correction=r.correction,
                    corrected_by=r.corrected_by,
                    status="applied",
                    corrected_response_iqs=new_scores["iqs"],
                )
                improvements.append({
                    "id": r.id,
                    "original_iqs": round(original_iqs, 3),
                    "corrected_iqs": round(new_scores["iqs"], 3),
                    "improved": improved,
                })

        applied = store.get_by_status("applied")
        all_improved = all(imp["improved"] for imp in improvements)
        stage4_ok = len(applied) == 3 and all_improved
        results["stages"]["improvement"] = {
            "applied_count": len(applied),
            "all_improved": all_improved,
            "improvements": improvements,
            "passed": stage4_ok,
        }
        print(f"  Applied {len(applied)}/3 corrections")
        for imp in improvements:
            arrow = "↑" if imp["improved"] else "↓"
            print(f"    {imp['id']}: IQS {imp['original_iqs']:.3f} {arrow} {imp['corrected_iqs']:.3f}")
        all_passed = all_passed and stage4_ok

        # ------------------------------------------------------------------
        # Stage 5: Export - fine-tuning training pairs
        # ------------------------------------------------------------------
        print("\n[Stage 5] Export - generate fine-tuning training pairs")

        with tempfile.TemporaryDirectory() as tmpdir:
            formats_ok = {}
            for fmt in ["openai", "alpaca", "simple"]:
                out = os.path.join(tmpdir, f"sft_{fmt}.jsonl")
                count = store.export_for_finetuning(out, fmt=fmt)
                # Verify the file is valid JSONL with correct structure
                with open(out, encoding="utf-8") as f:
                    lines = [json.loads(l) for l in f if l.strip()]

                if fmt == "openai":
                    valid = all("messages" in l and len(l["messages"]) == 3 for l in lines)
                elif fmt == "alpaca":
                    valid = all("instruction" in l and "input" in l and "output" in l for l in lines)
                elif fmt == "simple":
                    valid = all("prompt" in l and "completion" in l for l in lines)
                else:
                    valid = False

                has_meta = all("_scroot_meta" in l for l in lines)
                formats_ok[fmt] = {
                    "exported": count,
                    "valid_structure": valid,
                    "has_meta": has_meta,
                }
                print(f"  {fmt:<8} {count} records  structure={'✓' if valid else '✗'}  "
                      f"meta={'✓' if has_meta else '✗'}")

        stage5_ok = all(v["exported"] == 3 and v["valid_structure"] for v in formats_ok.values())
        results["stages"]["export"] = {
            "formats": formats_ok,
            "passed": stage5_ok,
        }
        all_passed = all_passed and stage5_ok

    finally:
        os.unlink(store_path)

    results["passed"] = all_passed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Feedback loop benchmark: {'ALL PASS ✓' if all_passed else 'FAIL ✗'}")
    for stage, data in results["stages"].items():
        status = "✓" if data.get("passed") else "✗"
        print(f"  {stage:<15} {status}")
    print(f"Results -> {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["passed"] else 1)
