"""Tests for the typer CLI app (scroot.cli.app)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from scroot.cli import app  # noqa: E402

runner = CliRunner()


class TestDownloadModelCommand:
    def test_invokes_download(self):
        with patch("scroot.cli.download.download_model") as dl:
            result = runner.invoke(app, ["download-model", "--model", "smollm3"])
        assert result.exit_code == 0
        dl.assert_called_once_with("smollm3")

    def test_default_model(self):
        with patch("scroot.cli.download.download_model") as dl:
            result = runner.invoke(app, ["download-model"])
        assert result.exit_code == 0
        dl.assert_called_once_with("phi4-mini")

    def test_error_exits_nonzero(self):
        with patch(
            "scroot.cli.download.download_model",
            side_effect=RuntimeError("disk full"),
        ):
            result = runner.invoke(app, ["download-model"])
        assert result.exit_code == 1
        assert "ERROR" in result.output
        assert "disk full" in result.output


class TestModelInfoCommand:
    def test_invokes_print_model_info(self):
        with patch("scroot.cli.model_info.print_model_info") as info:
            result = runner.invoke(app, ["model-info"])
        assert result.exit_code == 0
        info.assert_called_once()


class TestScoreCommand:
    def _fake_result(self):
        from scroot import EntailmentResult

        return EntailmentResult(
            groundedness=0.91,
            completeness=0.80,
            relevance=0.75,
            consistency=0.95,
            confidence=0.60,
            iqs=0.81,
            flags=[],
        )

    def test_score_summary_output(self):
        with patch("scroot.score", return_value=self._fake_result()) as fn:
            result = runner.invoke(
                app,
                ["score", "--query", "Q?", "--response", "A.", "--context", "ctx"],
            )
        assert result.exit_code == 0
        assert "IQS:          0.81" in result.output
        assert "Groundedness: 0.91" in result.output
        fn.assert_called_once_with(query="Q?", response="A.", context=["ctx"])

    def test_score_without_context(self):
        fake = self._fake_result()
        fake.groundedness = None
        with patch("scroot.score", return_value=fake) as fn:
            result = runner.invoke(app, ["score", "--query", "Q?", "--response", "A."])
        assert result.exit_code == 0
        assert "Groundedness: n/a (no context provided)" in result.output
        fn.assert_called_once_with(query="Q?", response="A.", context=None)

    def test_score_json_output(self):
        with patch("scroot.score", return_value=self._fake_result()):
            result = runner.invoke(
                app, ["score", "--query", "Q?", "--response", "A.", "--json"]
            )
        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert data["iqs"] == 0.81


class TestServeCommand:
    def test_serve_starts_uvicorn(self):
        fake_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            with patch(
                "scroot.dashboard.server.create_app", return_value="fake-app"
            ) as create:
                result = runner.invoke(
                    app, ["serve", "--port", "9999", "--store", "./s.jsonl"]
                )
        assert result.exit_code == 0
        assert "SCROOT Review Console" in result.output
        assert "9999" in result.output
        create.assert_called_once_with(
            store_path="./s.jsonl", hosted=False, host="127.0.0.1", auth_token=None
        )
        fake_uvicorn.run.assert_called_once()
        _, kwargs = fake_uvicorn.run.call_args
        assert kwargs["port"] == 9999

    def test_serve_without_uvicorn_exits_with_hint(self):
        with patch.dict(sys.modules, {"uvicorn": None}):
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "scroot[dashboard]" in result.output
