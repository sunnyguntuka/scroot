"""Tests for dashboard hardening: key masking (H-1), base_url allowlist (M-2),
config file permissions (M-1), and token auth + bind warning (H-2)."""
from __future__ import annotations

import os
import stat
import sys

import pytest
from fastapi.testclient import TestClient

from scroot.dashboard.security import (
    ALLOWED_LLM_HOSTS,
    extract_request_token,
    is_loopback_host,
    mask_api_key,
    resolve_dashboard_token,
    token_matches,
    validate_base_url,
)


# ─── H-1: mask_api_key ────────────────────────────────────────────────────

class TestMaskApiKey:
    def test_empty_returns_empty(self):
        assert mask_api_key("") == ""
        assert mask_api_key(None) == ""

    def test_short_key_fully_hidden(self):
        assert mask_api_key("sk-12345") == "…"

    def test_long_key_shows_only_ends(self):
        masked = mask_api_key("sk-abcdefghijklmnopqrstuvwxyz")
        assert masked == "sk-a…wxyz"
        assert "bcdefghij" not in masked

    def test_never_returns_full_key(self):
        key = "sk-proj-supersecretvalue1234567890"
        assert key not in mask_api_key(key)


# ─── M-2: validate_base_url ───────────────────────────────────────────────

class TestValidateBaseUrl:
    def test_empty_is_allowed(self):
        validate_base_url("")
        validate_base_url(None)

    @pytest.mark.parametrize("host", sorted(ALLOWED_LLM_HOSTS))
    def test_known_providers_allowed(self, host):
        validate_base_url(f"https://{host}/v1")

    def test_localhost_allowed_for_ollama(self):
        validate_base_url("http://localhost:11434")
        validate_base_url("http://127.0.0.1:11434/v1")

    def test_arbitrary_host_rejected(self):
        with pytest.raises(ValueError):
            validate_base_url("https://evil.example.com/v1")

    def test_internal_metadata_address_rejected(self):
        with pytest.raises(ValueError):
            validate_base_url("http://169.254.169.254/latest/meta-data/")

    def test_private_host_rejected(self):
        with pytest.raises(ValueError):
            validate_base_url("http://10.0.0.5:8080/v1")

    def test_non_http_scheme_rejected(self):
        with pytest.raises(ValueError):
            validate_base_url("file:///etc/passwd")

    def test_localhost_blocked_when_allow_local_false(self):
        with pytest.raises(ValueError):
            validate_base_url("http://localhost:11434", allow_local=False)

    def test_override_env_allows_any_host(self, monkeypatch):
        monkeypatch.setenv("SCROOT_ALLOW_ANY_BASE_URL", "1")
        validate_base_url("https://my-internal-gateway.corp/v1")


# ─── H-2: bind-host + token helpers ───────────────────────────────────────

class TestLoopbackAndToken:
    def test_loopback_hosts(self):
        assert is_loopback_host("127.0.0.1")
        assert is_loopback_host("localhost")
        assert is_loopback_host("::1")
        assert is_loopback_host("")

    def test_non_loopback_hosts(self):
        assert not is_loopback_host("0.0.0.0")
        assert not is_loopback_host("192.168.1.10")

    def test_resolve_token_precedence(self, monkeypatch):
        monkeypatch.delenv("SCROOT_DASHBOARD_TOKEN", raising=False)
        assert resolve_dashboard_token(None) is None
        assert resolve_dashboard_token("explicit") == "explicit"
        monkeypatch.setenv("SCROOT_DASHBOARD_TOKEN", "from-env")
        assert resolve_dashboard_token(None) == "from-env"
        assert resolve_dashboard_token("explicit") == "explicit"  # explicit wins

    def test_token_matches_constant_time(self):
        assert token_matches("secret", "secret")
        assert not token_matches("wrong", "secret")
        assert not token_matches(None, "secret")
        assert not token_matches("", "secret")

    def test_extract_request_token_bearer(self):
        assert extract_request_token({"authorization": "Bearer abc123"}) == "abc123"

    def test_extract_request_token_custom_header(self):
        assert extract_request_token({"x-scroot-token": "xyz"}) == "xyz"

    def test_extract_request_token_absent(self):
        assert extract_request_token({}) is None


# ─── M-1: config file permissions (POSIX only) ────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
class TestConfigFilePermissions:
    def test_corrector_config_written_0600(self, tmp_path):
        from scroot.config.corrector import APIConfig, CorrectorConfig

        path = tmp_path / "sub" / "config.json"
        cc = CorrectorConfig(mode="api", api=APIConfig(api_key="sk-secret"))
        cc.save(path)

        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"


# ─── H-2: token auth middleware via create_app ────────────────────────────

class TestTokenAuthMiddleware:
    def _client(self, tmp_path, token=None, host="127.0.0.1"):
        from scroot.dashboard.server import create_app

        store_path = str(tmp_path / "store.jsonl")
        app = create_app(store_path=store_path, host=host, auth_token=token)
        return TestClient(app)

    def test_no_token_configured_allows_api(self, tmp_path):
        client = self._client(tmp_path, token=None)
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/settings").status_code == 200

    def test_token_required_rejects_missing(self, tmp_path):
        client = self._client(tmp_path, token="s3cret")
        assert client.get("/api/settings").status_code == 401

    def test_token_required_accepts_bearer(self, tmp_path):
        client = self._client(tmp_path, token="s3cret")
        r = client.get("/api/settings", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200

    def test_token_required_accepts_custom_header(self, tmp_path):
        client = self._client(tmp_path, token="s3cret")
        r = client.get("/api/settings", headers={"X-Scroot-Token": "s3cret"})
        assert r.status_code == 200

    def test_token_required_rejects_wrong(self, tmp_path):
        client = self._client(tmp_path, token="s3cret")
        r = client.get("/api/settings", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_health_bypasses_token(self, tmp_path):
        client = self._client(tmp_path, token="s3cret")
        assert client.get("/api/health").status_code == 200

    def test_non_loopback_bind_without_token_warns(self, tmp_path):
        from scroot.dashboard.server import create_app

        store_path = str(tmp_path / "store.jsonl")
        with pytest.warns(UserWarning, match="non-loopback"):
            create_app(store_path=store_path, host="0.0.0.0")

    def test_non_loopback_bind_with_token_no_warning(self, tmp_path, recwarn):
        from scroot.dashboard.server import create_app

        store_path = str(tmp_path / "store.jsonl")
        create_app(store_path=store_path, host="0.0.0.0", auth_token="s3cret")
        assert not any("non-loopback" in str(w.message) for w in recwarn.list)


def _ui_built() -> bool:
    from pathlib import Path

    from scroot.dashboard.server import UI_DIST_PATH

    return (Path(UI_DIST_PATH) / "index.html").exists()


@pytest.mark.skipif(not _ui_built(), reason="UI not built (run `npm run build`)")
class TestSpaFallback:
    """The dashboard uses BrowserRouter; the server must serve index.html for
    client-side routes (deep links / refresh) and still 404 unknown API paths."""

    def _client(self, tmp_path):
        from scroot.dashboard.server import create_app

        return TestClient(create_app(store_path=str(tmp_path / "store.jsonl")))

    def test_deep_link_serves_index_html(self, tmp_path):
        client = self._client(tmp_path)
        r = client.get("/queue")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_nested_deep_link_serves_index_html(self, tmp_path):
        client = self._client(tmp_path)
        assert client.get("/queue/rec_001").status_code == 200

    def test_unknown_api_path_still_404(self, tmp_path):
        client = self._client(tmp_path)
        assert client.get("/api/does-not-exist").status_code == 404

    def test_root_serves_index_html(self, tmp_path):
        client = self._client(tmp_path)
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
