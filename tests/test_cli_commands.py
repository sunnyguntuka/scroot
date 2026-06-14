"""Tests for scroot CLI commands: download-model and model-info."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scroot.cli.download import download_model
from scroot.cli.model_info import print_model_info


class TestDownloadModel:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            download_model("nonexistent-model-xyz")

    def test_already_downloaded_skips_download(self, capsys):
        with patch("scroot.cli.download.is_model_downloaded", return_value=True):
            with patch("scroot.cli.download.get_model_path") as mock_path:
                mock_path.return_value = Path("/fake/path.gguf")
                download_model("phi4-mini")
        captured = capsys.readouterr()
        assert "already downloaded" in captured.out

    def _mock_hf(self):
        """Return a MagicMock that stands in for huggingface_hub."""
        mock_hf = MagicMock()
        mock_hf.hf_hub_download = MagicMock()
        return mock_hf

    def test_no_huggingface_hub_raises(self):
        with patch("scroot.cli.download.is_model_downloaded", return_value=False):
            with patch.dict(sys.modules, {"huggingface_hub": None}):
                with pytest.raises(RuntimeError, match="huggingface-hub"):
                    download_model("phi4-mini")

    def test_success_calls_hf_download(self, tmp_path, capsys):
        mock_hf = self._mock_hf()
        with patch("scroot.cli.download.is_model_downloaded", return_value=False):
            with patch("scroot.cli.download.get_model_dir", return_value=tmp_path):
                with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                    download_model("phi4-mini")
        mock_hf.hf_hub_download.assert_called_once()
        captured = capsys.readouterr()
        assert "Downloading" in captured.out

    def test_success_output_mentions_model_name(self, tmp_path, capsys):
        mock_hf = self._mock_hf()
        with patch("scroot.cli.download.is_model_downloaded", return_value=False):
            with patch("scroot.cli.download.get_model_dir", return_value=tmp_path):
                with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                    download_model("phi4-mini")
        captured = capsys.readouterr()
        assert "Qwen2.5-3B-Instruct" in captured.out

    def test_smollm3_downloads(self, tmp_path, capsys):
        mock_hf = self._mock_hf()
        with patch("scroot.cli.download.is_model_downloaded", return_value=False):
            with patch("scroot.cli.download.get_model_dir", return_value=tmp_path):
                with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                    download_model("smollm3")
        mock_hf.hf_hub_download.assert_called_once()

    def test_hf_download_passes_resume_flag(self, tmp_path):
        mock_hf = self._mock_hf()
        with patch("scroot.cli.download.is_model_downloaded", return_value=False):
            with patch("scroot.cli.download.get_model_dir", return_value=tmp_path):
                with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                    download_model("phi4-mini")
        kwargs = mock_hf.hf_hub_download.call_args[1]
        assert kwargs.get("resume_download") is True

    def test_default_model_is_phi4_mini(self, tmp_path, capsys):
        mock_hf = self._mock_hf()
        with patch("scroot.cli.download.is_model_downloaded", return_value=False):
            with patch("scroot.cli.download.get_model_dir", return_value=tmp_path):
                with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                    download_model()  # no model_id arg → uses DEFAULT_MODEL_ID
        mock_hf.hf_hub_download.assert_called_once()


class TestPrintModelInfo:
    def test_runs_without_error(self, capsys):
        print_model_info()
        captured = capsys.readouterr()
        assert "scroot models" in captured.out
        assert "Qwen2.5-3B-Instruct" in captured.out

    def test_lists_both_models(self, capsys):
        print_model_info()
        captured = capsys.readouterr()
        assert "Qwen2.5-3B-Instruct" in captured.out
        assert "Qwen2.5-1.5B-Instruct" in captured.out

    def test_shows_storage_path(self, capsys):
        print_model_info()
        captured = capsys.readouterr()
        assert "scroot" in captured.out.lower()

    def test_shows_download_hint(self, capsys):
        print_model_info()
        captured = capsys.readouterr()
        assert "download-model" in captured.out

    def test_marks_default_model(self, capsys):
        print_model_info()
        captured = capsys.readouterr()
        assert "default" in captured.out.lower()

    def test_shows_license_info(self, capsys):
        print_model_info()
        captured = capsys.readouterr()
        assert "MIT" in captured.out or "Apache" in captured.out
