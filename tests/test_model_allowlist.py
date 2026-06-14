"""Tests for model allowlist (C-2)."""

import pytest
from scroot.models import (
    validate_model_name,
    trust_model,
    get_embedding_model,
    DEFAULT_ALLOWED_MODELS,
    _allowed_models,
)


def test_default_allowed_models_present():
    assert "all-MiniLM-L6-v2" in DEFAULT_ALLOWED_MODELS
    assert "cross-encoder/nli-deberta-v3-base" in DEFAULT_ALLOWED_MODELS


def test_allowed_model_passes_validation():
    # Should not raise
    validate_model_name("all-MiniLM-L6-v2")


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="not on the trusted allowlist"):
        validate_model_name("attacker/malicious-model")


def test_trust_model_expands_allowlist():
    test_name = "test-org/custom-model-xyz-unique"
    assert test_name not in DEFAULT_ALLOWED_MODELS
    trust_model(test_name)
    validate_model_name(test_name)  # should not raise now
    # cleanup
    _allowed_models.discard(test_name)


@pytest.mark.needs_model
def test_pre_instantiated_embedding_bypasses_validation():
    model = get_embedding_model("all-MiniLM-L6-v2")
    # Passing the instance back in skips allowlist check entirely
    result = get_embedding_model(model)
    assert result is model


def test_unknown_model_error_message_includes_trust_hint():
    with pytest.raises(ValueError, match="trust_model"):
        validate_model_name("some/unknown-model")
