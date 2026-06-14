"""Deduplication threshold tests."""

from datetime import datetime, timezone

import pytest

from scroot.context import dedup as dedup_mod
from scroot.context.dedup import deduplicate
from scroot.context.payload import ContextEntry


def make_entry(content, source="retrieval", weight=0.85):
    return ContextEntry(
        source=source,
        content=content,
        added_at=datetime.now(timezone.utc),
        metadata={},
        source_weight=weight,
        token_count=len(content) // 4 + 1,
        was_scrubbed=False,
        scrub_summary={},
    )


class TestExactDedup:
    def test_exact_duplicates_removed(self):
        entries = [make_entry("Refunds within 30 days."),
                   make_entry("Refunds within 30 days.")]
        assert len(deduplicate(entries)) == 1

    def test_whitespace_case_normalised(self):
        entries = [make_entry("Refunds  Within 30 Days."),
                   make_entry("refunds within 30 days.")]
        assert len(deduplicate(entries)) == 1

    def test_distinct_content_kept(self):
        entries = [make_entry("Refunds within 30 days."),
                   make_entry("Shipping takes 5 business days.")]
        assert len(deduplicate(entries)) == 2

    def test_first_occurrence_kept(self):
        first = make_entry("Refunds within 30 days.", source="reranker", weight=1.0)
        second = make_entry("Refunds within 30 days.", source="retrieval", weight=0.85)
        kept = deduplicate([first, second])
        assert kept == [first]

    def test_empty_and_single(self):
        assert deduplicate([]) == []
        single = [make_entry("only one")]
        assert deduplicate(single) == single


class TestNearDedupFallback:
    """Force the non-embedding fallback path so tests stay model-free."""

    @pytest.fixture(autouse=True)
    def no_embeddings(self, monkeypatch):
        monkeypatch.setattr(dedup_mod, "_similarity_matrix", lambda *a, **k: None)

    def test_near_identical_merged(self):
        entries = [
            make_entry("All customers are eligible for a 30-day full refund at no cost."),
            make_entry("All customers are eligible for a 30-day full refund at no cost!!"),
        ]
        assert len(deduplicate(entries, threshold=0.92)) == 1

    def test_different_content_survives(self):
        entries = [
            make_entry("All customers are eligible for a 30-day full refund."),
            make_entry("Our headquarters are located in Toronto, Canada."),
        ]
        assert len(deduplicate(entries, threshold=0.92)) == 2

    def test_threshold_respected(self):
        # ~85% similar strings: dropped at low threshold, kept at high one
        a = "The quick brown fox jumps over the lazy dog near the river bank."
        b = "The quick brown fox jumps over the lazy cat near the river bank."
        entries = [make_entry(a), make_entry(b)]
        assert len(deduplicate(entries, threshold=0.50)) == 1
        assert len(deduplicate(entries, threshold=0.999)) == 2


@pytest.mark.needs_model
class TestNearDedupEmbeddings:
    def test_paraphrase_merged_with_embeddings(self):
        entries = [
            make_entry("Customers can get a full refund within 30 days of purchase."),
            make_entry("Customers can get a complete refund within 30 days of purchase."),
        ]
        assert len(deduplicate(entries, threshold=0.92)) == 1

    def test_unrelated_kept_with_embeddings(self):
        entries = [
            make_entry("Customers can get a full refund within 30 days."),
            make_entry("The mitochondria is the powerhouse of the cell."),
        ]
        assert len(deduplicate(entries, threshold=0.92)) == 2
