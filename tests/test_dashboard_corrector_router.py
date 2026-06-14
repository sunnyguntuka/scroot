"""Tests for /api/corrector endpoints via FastAPI TestClient."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scroot.dashboard.routers.corrector import corrector_router

app = FastAPI()
app.include_router(corrector_router(), prefix="/api/corrector")
client = TestClient(app)


class TestListModels:
    def test_returns_200(self):
        resp = client.get("/api/corrector/models")
        assert resp.status_code == 200

    def test_contains_models_key(self):
        data = client.get("/api/corrector/models").json()
        assert "models" in data

    def test_has_phi4_and_smollm3(self):
        models = client.get("/api/corrector/models").json()["models"]
        ids = [m["id"] for m in models]
        assert "phi4-mini" in ids
        assert "smollm3" in ids

    def test_model_has_required_fields(self):
        models = client.get("/api/corrector/models").json()["models"]
        for m in models:
            assert "id" in m
            assert "name" in m
            assert "size_gb" in m
            assert "downloaded" in m
            assert "is_default" in m
            assert "license" in m

    def test_phi4_is_default(self):
        models = client.get("/api/corrector/models").json()["models"]
        phi4 = next(m for m in models if m["id"] == "phi4-mini")
        assert phi4["is_default"] is True

    def test_not_downloaded_when_file_absent(self):
        models = client.get("/api/corrector/models").json()["models"]
        for m in models:
            if not m["downloaded"]:
                assert m["path"] is None


class TestStartDownload:
    def test_unknown_model_returns_404(self):
        resp = client.post("/api/corrector/models/nonexistent/download")
        assert resp.status_code == 404

    def test_already_downloaded_returns_ready(self):
        with patch("scroot.dashboard.routers.corrector.is_model_downloaded", return_value=True):
            resp = client.post("/api/corrector/models/phi4-mini/download")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_starts_download_thread(self):
        with patch("scroot.dashboard.routers.corrector.is_model_downloaded", return_value=False):
            with patch("scroot.dashboard.routers.corrector._do_download"):
                with patch("threading.Thread") as mock_thread:
                    instance = MagicMock()
                    mock_thread.return_value = instance
                    resp = client.post("/api/corrector/models/phi4-mini/download")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "phi4-mini"
        assert data["status"] == "downloading"

    def test_duplicate_download_returns_downloading(self):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads["smollm3"] = {"status": "downloading"}
        with patch("scroot.dashboard.routers.corrector.is_model_downloaded", return_value=False):
            resp = client.post("/api/corrector/models/smollm3/download")
        assert resp.status_code == 200
        assert resp.json()["status"] == "downloading"
        del mod._downloads["smollm3"]


class TestDownloadStatus:
    def setup_method(self):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads.pop("phi4-mini", None)
        mod._downloads.pop("smollm3", None)

    def test_unknown_model_returns_404(self):
        resp = client.get("/api/corrector/models/nonexistent/download-status")
        assert resp.status_code == 404

    def test_already_downloaded_no_state_returns_ready(self):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads.pop("phi4-mini", None)
        with patch("scroot.dashboard.routers.corrector.is_model_downloaded", return_value=True):
            resp = client.get("/api/corrector/models/phi4-mini/download-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["progress_pct"] == 100

    def test_downloading_returns_progress(self):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads["phi4-mini"] = {
            "status": "downloading",
            "progress_bytes": 500_000_000,
            "total_bytes": 2_400_000_000,
            "progress_pct": 20,
            "eta_seconds": 120,
            "error": None,
        }
        with patch("scroot.dashboard.routers.corrector.is_model_downloaded", return_value=False):
            resp = client.get("/api/corrector/models/phi4-mini/download-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "downloading"
        assert data["progress_pct"] == 20
        assert data["eta_seconds"] == 120

    def test_failed_download_returns_error(self):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads["phi4-mini"] = {
            "status": "failed",
            "progress_bytes": 0,
            "total_bytes": 0,
            "progress_pct": 0,
            "eta_seconds": None,
            "error": "Network timeout",
        }
        with patch("scroot.dashboard.routers.corrector.is_model_downloaded", return_value=False):
            resp = client.get("/api/corrector/models/phi4-mini/download-status")
        data = resp.json()
        assert data["status"] == "failed"
        assert "Network timeout" in data["error"]

    def test_no_state_returns_pending(self):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads.pop("phi4-mini", None)
        with patch("scroot.dashboard.routers.corrector.is_model_downloaded", return_value=False):
            resp = client.get("/api/corrector/models/phi4-mini/download-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"


class TestDeleteModel:
    def test_unknown_model_returns_404(self):
        resp = client.delete("/api/corrector/models/nonexistent")
        assert resp.status_code == 404

    def test_not_downloaded_returns_404(self):
        with patch("scroot.dashboard.routers.corrector.get_model_path") as mp:
            mock_p = MagicMock()
            mock_p.exists.return_value = False
            mp.return_value = mock_p
            resp = client.delete("/api/corrector/models/phi4-mini")
        assert resp.status_code == 404

    def test_deletes_file_and_returns_freed_gb(self, tmp_path):
        fake_file = tmp_path / "model.gguf"
        fake_file.write_bytes(b"0" * 1024 * 1024)  # 1 MB

        with patch("scroot.dashboard.routers.corrector.get_model_path") as mp:
            mp.return_value = fake_file
            resp = client.delete("/api/corrector/models/phi4-mini")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["model_id"] == "phi4-mini"
        assert data["freed_gb"] >= 0
        assert not fake_file.exists()

    def test_unloads_active_local_corrector(self, tmp_path):
        fake_file = tmp_path / "model.gguf"
        fake_file.write_bytes(b"x")

        mock_corrector = MagicMock()
        mock_corrector.__class__ = MagicMock

        with patch("scroot.dashboard.routers.corrector.get_model_path", return_value=fake_file):
            with patch("scroot.dashboard.routers.corrector._active_corrector", mock_corrector, create=True):
                resp = client.delete("/api/corrector/models/phi4-mini")
        assert resp.status_code == 200


class TestTestCorrector:
    def test_disabled_returns_disabled(self):
        from scroot.config.corrector import CorrectorConfig
        with patch("scroot.dashboard.routers.corrector.CorrectorConfig.load") as mock_load:
            mock_load.return_value = CorrectorConfig(mode="disabled")
            resp = client.post("/api/corrector/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "disabled"
        assert data["error"] == "Corrector is disabled"

    def test_unavailable_corrector_returns_error(self):
        from scroot.config.corrector import CorrectorConfig
        mock_corrector = MagicMock()
        mock_corrector.is_available = False
        with patch("scroot.dashboard.routers.corrector.CorrectorConfig.load") as mock_load:
            mock_load.return_value = CorrectorConfig(mode="api")
            with patch("scroot.corrector.get_corrector", return_value=mock_corrector):
                resp = client.post("/api/corrector/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is not None

    def test_local_corrector_success(self):
        from scroot.config.corrector import CorrectorConfig

        cfg = CorrectorConfig(mode="local")
        mock_corrector = MagicMock()
        mock_corrector.is_available = True
        mock_corrector.draft_correction.return_value = "Paris is the capital of France."
        mock_corrector.tok_per_sec.return_value = 15.0

        with patch("scroot.dashboard.routers.corrector.CorrectorConfig.load", return_value=cfg):
            with patch("scroot.corrector.get_corrector", return_value=mock_corrector):
                resp = client.post("/api/corrector/test")

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert "Paris" in (data["sample_output"] or "")

    def test_api_corrector_success(self):
        from scroot.config.corrector import APIConfig, CorrectorConfig

        cfg = CorrectorConfig(mode="api", api=APIConfig(api_key="sk-test", model="gpt-4o-mini"))
        mock_corrector = MagicMock()
        mock_corrector.is_available = True
        mock_corrector.draft_correction.return_value = "Paris."

        with patch("scroot.dashboard.routers.corrector.CorrectorConfig.load", return_value=cfg):
            with patch("scroot.corrector.get_corrector", return_value=mock_corrector):
                resp = client.post("/api/corrector/test")

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "api"
        assert data["model"] == "gpt-4o-mini"

    def test_corrector_exception_captured(self):
        from scroot.config.corrector import CorrectorConfig

        cfg = CorrectorConfig(mode="api")
        mock_corrector = MagicMock()
        mock_corrector.is_available = True
        mock_corrector.draft_correction.side_effect = RuntimeError("API timeout")

        with patch("scroot.dashboard.routers.corrector.CorrectorConfig.load", return_value=cfg):
            with patch("scroot.corrector.get_corrector", return_value=mock_corrector):
                resp = client.post("/api/corrector/test")

        assert resp.status_code == 200
        data = resp.json()
        assert "API timeout" in (data["error"] or "")


class TestDoDownload:
    def test_do_download_no_huggingface_hub(self):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads["phi4-mini"] = {
            "status": "pending", "progress_bytes": 0,
            "total_bytes": 0, "progress_pct": 0,
            "eta_seconds": None, "error": None, "_started": 0,
        }
        with patch.dict(__import__("sys").modules, {"huggingface_hub": None}):
            mod._do_download("phi4-mini")
        assert mod._downloads["phi4-mini"]["status"] == "failed"
        assert "huggingface-hub" in mod._downloads["phi4-mini"]["error"]
        del mod._downloads["phi4-mini"]

    def test_do_download_exception_sets_failed(self, tmp_path):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads["phi4-mini"] = {
            "status": "pending", "progress_bytes": 0,
            "total_bytes": 0, "progress_pct": 0,
            "eta_seconds": None, "error": None, "_started": 0,
        }
        with patch("scroot.dashboard.routers.corrector.get_model_path") as mp:
            mp.return_value = tmp_path / "phi4-mini" / "model.gguf"
            mock_hf = MagicMock()
            mock_hf.hf_hub_download.side_effect = RuntimeError("Download failed")
            with patch.dict(__import__("sys").modules, {"huggingface_hub": mock_hf}):
                mod._do_download("phi4-mini")
        assert mod._downloads["phi4-mini"]["status"] == "failed"
        del mod._downloads["phi4-mini"]

    def test_do_download_success_sets_ready(self, tmp_path):
        import scroot.dashboard.routers.corrector as mod
        mod._downloads["phi4-mini"] = {
            "status": "pending", "progress_bytes": 0,
            "total_bytes": 0, "progress_pct": 0,
            "eta_seconds": None, "error": None, "_started": 0,
        }
        with patch("scroot.dashboard.routers.corrector.get_model_path") as mp:
            mp.return_value = tmp_path / "phi4-mini" / "model.gguf"
            mock_hf = MagicMock()
            with patch.dict(__import__("sys").modules, {"huggingface_hub": mock_hf}):
                mod._do_download("phi4-mini")
        state = mod._downloads["phi4-mini"]
        assert state["status"] == "ready"
        assert state["progress_pct"] == 100
        assert state["eta_seconds"] == 0
        assert state["error"] is None
        kwargs = mock_hf.hf_hub_download.call_args[1]
        assert kwargs["token"] is False
        assert kwargs["resume_download"] is True
        del mod._downloads["phi4-mini"]


class TestRuntimeStatus:
    def test_returns_install_flags(self):
        resp = client.get("/api/corrector/runtime")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["llama_cpp_installed"], bool)
        assert isinstance(data["hf_hub_installed"], bool)
        assert data["ready"] == (
            data["llama_cpp_installed"] and data["hf_hub_installed"]
        )


class TestDeleteUnloadGuard:
    def test_delete_proceeds_when_unload_check_fails(self, tmp_path, monkeypatch):
        """The active-corrector unload guard swallows errors and still deletes."""
        import scroot.corrector as corrector_pkg

        model_file = tmp_path / "phi4-mini" / "model.gguf"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"x" * 1024)

        # Make `from scroot.corrector import _active_corrector` fail
        monkeypatch.delattr(corrector_pkg, "_active_corrector")

        with patch(
            "scroot.dashboard.routers.corrector.get_model_path",
            return_value=model_file,
        ):
            resp = client.delete("/api/corrector/models/phi4-mini")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert not model_file.exists()
