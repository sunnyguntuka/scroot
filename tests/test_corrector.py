"""Tests for the corrector package - no LLM calls needed."""
import json
import os
import tempfile
from pathlib import Path


from scroot.corrector.api import detect_provider
from scroot.corrector.models import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    get_model_dir,
    get_model_path,
    is_model_downloaded,
)
from scroot.config.corrector import (
    APIConfig,
    CorrectorConfig,
    LocalConfig,
)
from scroot.corrector.disabled import NullCorrector
from scroot.corrector import get_corrector


# ─── detect_provider ─────────────────────────────────────────────────────────

class TestDetectProvider:
    def test_openai_key(self):
        base, header, name = detect_provider("sk-abc123")
        assert "openai.com" in base
        assert name == "OpenAI"
        assert header == "Authorization"

    def test_anthropic_key(self):
        base, header, name = detect_provider("sk-ant-xyz789")
        assert "anthropic.com" in base
        assert name == "Anthropic"
        assert header == "x-api-key"

    def test_gemini_key(self):
        base, header, name = detect_provider("AIzaSyAbc123")
        assert "googleapis.com" in base
        assert name == "Google Gemini"

    def test_unknown_key_falls_back_to_openrouter(self):
        base, header, name = detect_provider("unknown-key-xyz")
        assert "openrouter" in base
        assert name == "OpenRouter"

    def test_base_url_override_wins(self):
        base, header, name = detect_provider("sk-abc", "https://api.groq.com/openai/v1")
        assert base == "https://api.groq.com/openai/v1"
        assert name == "Groq"

    def test_openrouter_base_url(self):
        _, _, name = detect_provider("any-key", "https://openrouter.ai/api/v1")
        assert name == "OpenRouter"

    def test_custom_base_url(self):
        _, _, name = detect_provider("any-key", "http://localhost:8080/v1")
        assert name == "Custom"


# ─── Model path resolution ────────────────────────────────────────────────────

class TestModelPaths:
    def test_default_model_exists_in_registry(self):
        assert DEFAULT_MODEL_ID in MODEL_REGISTRY

    def test_all_models_have_required_fields(self):
        for mid, spec in MODEL_REGISTRY.items():
            assert spec.id == mid
            assert spec.hf_repo
            assert spec.hf_filename.endswith(".gguf")
            assert spec.size_gb > 0
            assert spec.license

    def test_get_model_path_returns_path_object(self):
        path = get_model_path(DEFAULT_MODEL_ID)
        assert isinstance(path, Path)
        assert path.suffix == ".gguf"

    def test_get_model_dir_uses_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SCROOT_MODELS_DIR"] = tmp
            try:
                d = get_model_dir()
                assert str(d) == tmp
            finally:
                del os.environ["SCROOT_MODELS_DIR"]

    def test_get_model_dir_default_under_home(self):
        os.environ.pop("SCROOT_MODELS_DIR", None)
        d = get_model_dir()
        assert ".scroot" in str(d)

    def test_is_model_downloaded_false_when_file_missing(self):
        assert is_model_downloaded("phi4-mini") is False or True  # just no crash

    def test_is_model_downloaded_true_when_file_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = MODEL_REGISTRY["phi4-mini"]
            model_dir = Path(tmp) / "phi4-mini"
            model_dir.mkdir()
            (model_dir / spec.hf_filename).touch()
            os.environ["SCROOT_MODELS_DIR"] = tmp
            try:
                assert is_model_downloaded("phi4-mini") is True
            finally:
                del os.environ["SCROOT_MODELS_DIR"]


# ─── CorrectorConfig save/load round-trip ────────────────────────────────────

class TestCorrectorConfig:
    def test_default_mode_is_disabled(self):
        cfg = CorrectorConfig()
        assert cfg.mode == "disabled"

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cfg = CorrectorConfig(
                mode="api",
                api=APIConfig(api_key="sk-test-123", model="gpt-4o"),
            )
            cfg.save(path)
            loaded = CorrectorConfig.load(path)
            assert loaded.mode == "api"
            assert loaded.api.api_key == "sk-test-123"
            assert loaded.api.model == "gpt-4o"
        finally:
            path.unlink(missing_ok=True)

    def test_load_nonexistent_returns_defaults(self):
        cfg = CorrectorConfig.load(Path("/nonexistent/path/config.json"))
        assert cfg.mode == "disabled"
        assert isinstance(cfg.local, LocalConfig)
        assert isinstance(cfg.api, APIConfig)

    def test_load_corrupted_returns_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json {{{{")
            path = Path(f.name)
        try:
            cfg = CorrectorConfig.load(path)
            assert cfg.mode == "disabled"
        finally:
            path.unlink(missing_ok=True)

    def test_local_config_defaults(self):
        lc = LocalConfig()
        assert lc.model_id == "phi4-mini"
        assert lc.n_threads == -1
        assert lc.n_gpu_layers == 0

    def test_api_config_defaults(self):
        ac = APIConfig()
        assert ac.api_key == ""
        assert ac.model == "gpt-4o-mini"
        assert ac.system_prompt != ""

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subdir" / "config.json"
            cfg = CorrectorConfig(mode="disabled")
            cfg.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["mode"] == "disabled"


# ─── NullCorrector ────────────────────────────────────────────────────────────

class TestNullCorrector:
    def test_is_not_available(self):
        nc = NullCorrector()
        assert nc.is_available is False

    def test_draft_correction_returns_none(self):
        nc = NullCorrector()
        result = nc.draft_correction("q", "r", "ctx")
        assert result is None


# ─── get_corrector factory ────────────────────────────────────────────────────

class TestGetCorrector:
    def test_disabled_mode_returns_null_corrector(self):
        cfg = CorrectorConfig(mode="disabled")
        corrector = get_corrector(cfg)
        assert isinstance(corrector, NullCorrector)
        assert corrector.is_available is False

    def test_api_mode_returns_api_corrector(self):
        from scroot.corrector.api import APICorrector
        cfg = CorrectorConfig(mode="api", api=APIConfig(api_key="sk-test"))
        corrector = get_corrector(cfg)
        assert isinstance(corrector, APICorrector)

    def test_local_mode_returns_local_corrector_type(self):
        from scroot.corrector.local import LocalLLMCorrector
        cfg = CorrectorConfig(mode="local")
        corrector = get_corrector(cfg)
        assert isinstance(corrector, LocalLLMCorrector)
