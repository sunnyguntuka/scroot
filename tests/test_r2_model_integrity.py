"""Tests for R2: model integrity SHA-256 checking in preflight().

All tests use a fake HF cache directory (no real model downloads required).
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

import scroot.runtime as runtime_mod
from scroot.runtime import _HASH_CACHE, _find_weight_files, _sha256_file, preflight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_dir(hub: pathlib.Path, slug: str, weight_bytes: bytes) -> pathlib.Path:
    """Create a minimal HF-hub model directory with a single weight file."""
    model_dir = hub / f"models--{slug}"
    snapshot_dir = model_dir / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    weight_file = snapshot_dir / "model.safetensors"
    weight_file.write_bytes(weight_bytes)
    refs_dir = model_dir / "refs"
    refs_dir.mkdir()
    (refs_dir / "main").write_text("abc123", encoding="utf-8")
    return model_dir


def _setup_embedding_models(hub: pathlib.Path) -> None:
    """Create both embedding model paths required by preflight().

    preflight() checks for:
      sentence-transformers/all-MiniLM-L6-v2 → models--sentence-transformers--all-MiniLM-L6-v2
      all-MiniLM-L6-v2                        → models--all-MiniLM-L6-v2  (bare-name fallback)
    Both must exist for ready=True when no models are missing.
    """
    _make_model_dir(hub, "sentence-transformers--all-MiniLM-L6-v2", b"emb weights")
    _make_model_dir(hub, "all-MiniLM-L6-v2", b"emb weights")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def clear_hash_cache():
    """Clear the module-level hash cache before every test."""
    _HASH_CACHE.clear()
    yield
    _HASH_CACHE.clear()


@pytest.fixture()
def fake_hub(tmp_path):
    """Return a fake HF hub directory path (does not create any model dirs)."""
    hub = tmp_path / "hub"
    hub.mkdir()
    return tmp_path  # this is the HF_HOME; hub is at HF_HOME/hub


# ---------------------------------------------------------------------------
# TestFindWeightFiles
# ---------------------------------------------------------------------------

class TestFindWeightFiles:
    def test_finds_safetensors_via_refs_main(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, "cross-encoder--nli-deberta-v3-base", b"weights")
        files = _find_weight_files(model_dir)
        assert len(files) == 1
        assert files[0].name == "model.safetensors"

    def test_falls_back_to_latest_snapshot_dir(self, tmp_path):
        model_dir = tmp_path / "models--test"
        snap = model_dir / "snapshots" / "deadbeef"
        snap.mkdir(parents=True)
        (snap / "pytorch_model.bin").write_bytes(b"w")
        files = _find_weight_files(model_dir)
        assert files[0].name == "pytorch_model.bin"

    def test_empty_dir_returns_empty(self, tmp_path):
        model_dir = tmp_path / "models--empty"
        model_dir.mkdir()
        assert _find_weight_files(model_dir) == []


# ---------------------------------------------------------------------------
# TestSha256File
# ---------------------------------------------------------------------------

class TestSha256File:
    def test_correct_digest(self, tmp_path):
        data = b"hello world"
        f = tmp_path / "test.bin"
        f.write_bytes(data)
        assert _sha256_file(f) == _sha256_bytes(data)

    def test_cache_hit_skips_rehash(self, tmp_path, monkeypatch):
        data = b"cache me"
        f = tmp_path / "w.bin"
        f.write_bytes(data)

        # First call: populates cache
        digest = _sha256_file(f)
        assert len(_HASH_CACHE) == 1

        # Patch hashlib.sha256 to blow up — second call must use cache
        def _boom(*a, **kw):
            raise AssertionError("sha256 called again after caching")

        monkeypatch.setattr(hashlib, "sha256", _boom)
        assert _sha256_file(f) == digest  # hits cache, not hashlib


# ---------------------------------------------------------------------------
# TestPreflightIntegrity
# ---------------------------------------------------------------------------

class TestPreflightIntegrity:
    """Integrity checks with a fake HF cache — no model downloads needed."""

    def _setup_nli_model(self, hub: pathlib.Path, weight_bytes: bytes) -> str:
        """Create fake NLI model; return its expected SHA-256."""
        _make_model_dir(hub / "hub", "cross-encoder--nli-deberta-v3-base", weight_bytes)
        return _sha256_bytes(weight_bytes)

    def test_matching_hash_ok(self, fake_hub, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        weight_data = b"good weights"
        digest = self._setup_nli_model(fake_hub, weight_data)
        _setup_embedding_models(fake_hub / "hub")

        result = preflight(
            integrity="warn",
            expected_hashes={"cross-encoder/nli-deberta-v3-base": digest},
        )
        assert result["integrity"]["cross-encoder/nli-deberta-v3-base"] == "ok"
        assert result["ready"] is True

    def test_mismatch_warn_does_not_flip_ready(self, fake_hub, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        self._setup_nli_model(fake_hub, b"good weights")
        _setup_embedding_models(fake_hub / "hub")
        wrong_hash = _sha256_bytes(b"different bytes")

        result = preflight(
            integrity="warn",
            expected_hashes={"cross-encoder/nli-deberta-v3-base": wrong_hash},
        )
        assert result["integrity"]["cross-encoder/nli-deberta-v3-base"] == "mismatch"
        # warn: model is present → ready stays True (parity with no-integrity behavior)
        assert result["ready"] is True

    def test_mismatch_strict_sets_not_ready(self, fake_hub, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        self._setup_nli_model(fake_hub, b"good weights")
        wrong_hash = _sha256_bytes(b"different bytes")

        result = preflight(
            integrity="strict",
            expected_hashes={"cross-encoder/nli-deberta-v3-base": wrong_hash},
        )
        assert result["integrity"]["cross-encoder/nli-deberta-v3-base"] == "mismatch"
        assert result["ready"] is False

    def test_flipped_byte_detected(self, fake_hub, monkeypatch):
        """A single-byte change must register as mismatch."""
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        original = b"legitimate weights blob"
        digest = self._setup_nli_model(fake_hub, original)

        # Now flip one byte in the weight file
        model_dir = fake_hub / "hub" / "models--cross-encoder--nli-deberta-v3-base"
        weight_file = model_dir / "snapshots" / "abc123" / "model.safetensors"
        corrupted = bytearray(original)
        corrupted[0] ^= 0xFF
        weight_file.write_bytes(bytes(corrupted))

        result = preflight(
            integrity="strict",
            expected_hashes={"cross-encoder/nli-deberta-v3-base": digest},
        )
        assert result["integrity"]["cross-encoder/nli-deberta-v3-base"] == "mismatch"
        assert result["ready"] is False

    def test_unknown_model_not_in_manifest(self, fake_hub, monkeypatch):
        """Model present but no hash in manifest → 'unknown', warn leaves ready True."""
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        self._setup_nli_model(fake_hub, b"weights")
        _setup_embedding_models(fake_hub / "hub")

        result = preflight(integrity="warn")  # no expected_hashes, empty manifest
        assert result["integrity"]["cross-encoder/nli-deberta-v3-base"] == "unknown"
        assert result["ready"] is True

    def test_unknown_strict_sets_not_ready(self, fake_hub, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        self._setup_nli_model(fake_hub, b"weights")

        result = preflight(integrity="strict")
        assert result["integrity"]["cross-encoder/nli-deberta-v3-base"] == "unknown"
        assert result["ready"] is False

    def test_off_skips_integrity_key(self, fake_hub, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        self._setup_nli_model(fake_hub, b"weights")

        result = preflight(integrity="off")
        assert "integrity" not in result

    def test_cache_prevents_rehash(self, fake_hub, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        weight_data = b"cached weights"
        digest = self._setup_nli_model(fake_hub, weight_data)

        # First call: populates _HASH_CACHE
        preflight(
            integrity="warn",
            expected_hashes={"cross-encoder/nli-deberta-v3-base": digest},
        )
        assert len(_HASH_CACHE) == 1
        cache_key = next(iter(_HASH_CACHE))

        # Corrupt hashlib.sha256 — second call must not reach it
        def _boom(*a, **kw):
            raise AssertionError("hashlib.sha256 called on second preflight — cache miss!")

        monkeypatch.setattr(hashlib, "sha256", _boom)
        result2 = preflight(
            integrity="warn",
            expected_hashes={"cross-encoder/nli-deberta-v3-base": digest},
        )
        assert result2["integrity"]["cross-encoder/nli-deberta-v3-base"] == "ok"
        # Cache entry still has the same key
        assert cache_key in _HASH_CACHE

    def test_parity_warn_does_not_change_ready_missing_cache_dir(self, fake_hub, monkeypatch):
        """Default warn mode: ready/missing/cache_dir identical to integrity='off'."""
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        self._setup_nli_model(fake_hub, b"w")

        off_result = preflight(integrity="off")
        warn_result = preflight(integrity="warn")

        assert warn_result["ready"] == off_result["ready"]
        assert warn_result["missing"] == off_result["missing"]
        assert warn_result["cache_dir"] == off_result["cache_dir"]
        # warn adds the integrity key; off does not
        assert "integrity" in warn_result
        assert "integrity" not in off_result

    def test_expected_hashes_overrides_manifest(self, fake_hub, monkeypatch, tmp_path):
        """expected_hashes wins over bundled manifest when both have an entry."""
        monkeypatch.setenv("HF_HOME", str(fake_hub))
        weight_data = b"real weights"
        real_digest = self._setup_nli_model(fake_hub, weight_data)

        # Monkey-patch _load_manifest to return a wrong hash
        monkeypatch.setattr(runtime_mod, "_load_manifest", lambda: {
            "cross-encoder/nli-deberta-v3-base": _sha256_bytes(b"wrong"),
        })

        # expected_hashes has the correct hash — should win
        result = preflight(
            integrity="warn",
            expected_hashes={"cross-encoder/nli-deberta-v3-base": real_digest},
        )
        assert result["integrity"]["cross-encoder/nli-deberta-v3-base"] == "ok"
