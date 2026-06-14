from scroot.composite import compute_iqs


def test_all_perfect_scores():
    iqs = compute_iqs(1.0, 1.0, 1.0, 1.0, 1.0)
    assert iqs == 1.0


def test_one_low_score_penalized_harmonic():
    # Harmonic mean: one low score drives IQS well below 0.5
    iqs = compute_iqs(1.0, 0.1, 1.0, 1.0, 1.0, mode="harmonic")
    assert iqs < 0.5


def test_one_low_score_penalized_geometric():
    # Geometric mean: one low score still penalizes significantly
    # but not as severely as harmonic (gradual, not binary collapse)
    iqs_low = compute_iqs(1.0, 0.1, 1.0, 1.0, 1.0, mode="geometric")
    iqs_perfect = compute_iqs(1.0, 1.0, 1.0, 1.0, 1.0, mode="geometric")
    # Low score should penalize noticeably
    assert iqs_low < iqs_perfect * 0.8


def test_no_context_mode():
    # groundedness=None should still produce a valid score
    iqs = compute_iqs(None, 0.9, 0.9, 0.9, 0.9)
    assert 0.0 <= iqs <= 1.0


def test_no_context_vs_with_context():
    iqs_with = compute_iqs(0.9, 0.9, 0.9, 0.9, 0.9)
    iqs_without = compute_iqs(None, 0.9, 0.9, 0.9, 0.9)
    # Both should be close to 0.9
    assert abs(iqs_with - iqs_without) < 0.05


def test_zero_score_clamped():
    iqs = compute_iqs(0.0, 0.0, 0.0, 0.0, 0.0)
    assert iqs == 0.0


def test_custom_weights():
    # With all weight on relevance, iqs should track relevance closely
    weights = {"groundedness": 0.0, "completeness": 0.0, "relevance": 1.0, "consistency": 0.0, "confidence": 0.0}
    iqs = compute_iqs(0.9, 0.9, 0.5, 0.9, 0.9, weights=weights)
    # With all weight on relevance=0.5, iqs should be ~0.5
    assert abs(iqs - 0.5) < 0.01


def test_output_in_range():
    import random
    random.seed(42)
    for _ in range(20):
        scores = [random.random() for _ in range(5)]
        iqs = compute_iqs(*scores)
        assert 0.0 <= iqs <= 1.0
