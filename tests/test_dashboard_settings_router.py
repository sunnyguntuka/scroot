"""Tests for updated /api/settings endpoints via FastAPI TestClient."""
from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scroot.dashboard.routers.settings import settings_router

app = FastAPI()
app.include_router(settings_router(store=None), prefix="/api/settings")
client = TestClient(app)


class TestGetSettings:
    def test_returns_200(self):
        assert client.get("/api/settings").status_code == 200

    def test_contains_iqs_threshold(self):
        data = client.get("/api/settings").json()
        assert "iqs_threshold" in data
        assert isinstance(data["iqs_threshold"], float)

    def test_contains_metric_weights(self):
        data = client.get("/api/settings").json()
        assert "metric_weights" in data
        w = data["metric_weights"]
        assert "groundedness" in w
        assert "completeness" in w

    def test_contains_corrector_field(self):
        data = client.get("/api/settings").json()
        assert "corrector" in data

    def test_corrector_has_mode(self):
        data = client.get("/api/settings").json()
        assert "mode" in data["corrector"]

    def test_corrector_has_local_and_api(self):
        data = client.get("/api/settings").json()
        c = data["corrector"]
        assert "local" in c
        assert "api" in c

    def test_corrector_local_has_model_id(self):
        data = client.get("/api/settings").json()
        assert "model_id" in data["corrector"]["local"]

    def test_corrector_api_never_returns_full_key(self):
        data = client.get("/api/settings").json()
        api = data["corrector"]["api"]
        assert "api_key_set" in api
        assert "api_key" not in api  # full key must never appear

    def test_contains_legacy_llm_corrector(self):
        data = client.get("/api/settings").json()
        assert "llm_corrector" in data

    def test_legacy_llm_corrector_never_returns_full_key(self):
        # H-1: the legacy block must not echo the raw key back.
        client.put("/api/settings", json={
            "llm_corrector": {"provider": "openai", "api_key": "sk-supersecret-12345"}
        })
        lc = client.get("/api/settings").json()["llm_corrector"]
        assert "api_key" not in lc
        assert lc["api_key_set"] is True
        assert "supersecret" not in lc["api_key_hint"]

    def test_record_count_is_zero_with_no_store(self):
        data = client.get("/api/settings").json()
        assert data["record_count"] == 0


class TestUpdateSettings:
    def test_update_iqs_threshold(self):
        resp = client.put("/api/settings", json={"iqs_threshold": 0.80})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_update_metric_weights(self):
        weights = {"groundedness": 0.40, "completeness": 0.20,
                   "relevance": 0.20, "consistency": 0.10, "confidence": 0.10}
        resp = client.put("/api/settings", json={"metric_weights": weights})
        assert resp.status_code == 200

    def test_update_corrector_mode_disabled(self):
        resp = client.put("/api/settings", json={"corrector": {"mode": "disabled"}})
        assert resp.status_code == 200

    def test_update_corrector_mode_local(self):
        resp = client.put("/api/settings", json={
            "corrector": {"mode": "local", "local": {"model_id": "smollm3"}}
        })
        assert resp.status_code == 200

    def test_update_corrector_mode_api(self):
        resp = client.put("/api/settings", json={
            "corrector": {
                "mode": "api",
                "api": {"api_key": "sk-test-123", "model": "gpt-4o-mini"}
            }
        })
        assert resp.status_code == 200

    def test_update_corrector_api_persists_key(self, tmp_path):
        config_path = tmp_path / "config.json"
        with patch("scroot.dashboard.routers.settings.default_config_path", return_value=config_path):
            client.put("/api/settings", json={
                "corrector": {"mode": "api", "api": {"api_key": "sk-mykey", "model": "gpt-4o"}}
            })
            from scroot.config.corrector import CorrectorConfig
            loaded = CorrectorConfig.load(config_path)
        assert loaded.mode == "api"
        assert loaded.api.api_key == "sk-mykey"

    def test_update_legacy_llm_corrector(self):
        resp = client.put("/api/settings", json={
            "llm_corrector": {"provider": "openai", "api_key": "sk-x", "model": "gpt-4o-mini"}
        })
        assert resp.status_code == 200

    def test_clear_all_records_with_no_store_ok(self):
        resp = client.put("/api/settings", json={"clear_all_records": True})
        assert resp.status_code == 200

    def test_empty_body_ok(self):
        resp = client.put("/api/settings", json={})
        assert resp.status_code == 200

    def test_corrector_local_partial_update(self):
        resp = client.put("/api/settings", json={
            "corrector": {"mode": "local", "local": {"model_id": "phi4-mini", "n_threads": 4}}
        })
        assert resp.status_code == 200

    def test_corrector_api_with_base_url(self):
        resp = client.put("/api/settings", json={
            "corrector": {
                "mode": "api",
                "api": {"api_key": "sk-test", "base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3"}
            }
        })
        assert resp.status_code == 200

    def test_corrector_api_untrusted_base_url_rejected(self, tmp_path):
        # M-2: a non-provider base_url must be rejected with a clean 400.
        config_path = tmp_path / "config.json"
        with patch("scroot.dashboard.routers.settings.default_config_path", return_value=config_path):
            resp = client.put("/api/settings", json={
                "corrector": {
                    "mode": "api",
                    "api": {"api_key": "sk-x", "base_url": "http://169.254.169.254/", "model": "gpt-4o"}
                }
            })
        assert resp.status_code == 400

    def test_corrector_api_blank_key_preserves_existing(self, tmp_path):
        # H-1 write-only semantics: a blank api_key must not wipe the stored key.
        config_path = tmp_path / "config.json"
        with patch("scroot.dashboard.routers.settings.default_config_path", return_value=config_path):
            client.put("/api/settings", json={
                "corrector": {"mode": "api", "api": {"api_key": "sk-keepme", "model": "gpt-4o"}}
            })
            # Subsequent edit changes only the model, sends empty key.
            client.put("/api/settings", json={
                "corrector": {"mode": "api", "api": {"api_key": "", "model": "gpt-4o-mini"}}
            })
            from scroot.config.corrector import CorrectorConfig
            loaded = CorrectorConfig.load(config_path)
        assert loaded.api.api_key == "sk-keepme"
        assert loaded.api.model == "gpt-4o-mini"


class TestTestConnection:
    def test_no_provider_returns_error(self):
        resp = client.post("/api/settings/test-connection")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

    def test_returns_latency_ms(self):
        resp = client.post("/api/settings/test-connection")
        data = resp.json()
        assert "latency_ms" in data

    def test_returns_message(self):
        resp = client.post("/api/settings/test-connection")
        data = resp.json()
        assert "message" in data


class TestLegacyLlmJudgeRoutes:
    def test_get_llm_judge(self):
        resp = client.get("/api/settings/llm-judge")
        assert resp.status_code == 200
        data = resp.json()
        assert "provider" in data

    def test_put_llm_judge(self):
        resp = client.put("/api/settings/llm-judge", json={"provider": "openai", "model": "gpt-4o"})
        assert resp.status_code == 200

    def test_post_llm_judge_test(self):
        resp = client.post("/api/settings/llm-judge/test")
        assert resp.status_code == 200
