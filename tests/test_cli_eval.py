"""Tests for `scroot eval` (cli/eval.py + CLI command)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("yaml")
pytest.importorskip("typer")

from typer.testing import CliRunner  # noqa: E402

from scroot import EntailmentResult  # noqa: E402
from scroot.cli import app  # noqa: E402
from scroot.cli.eval import (  # noqa: E402
    EvalExample,
    EvalSuite,
    format_junit_xml,
    format_report,
    load_suite,
    run_suite,
)

runner = CliRunner()

FIXTURE = "tests/quality/support_regression.yaml"


def _result(iqs, groundedness=0.9):
    return EntailmentResult(
        groundedness=groundedness,
        completeness=0.9,
        relevance=0.9,
        consistency=0.9,
        confidence=0.9,
        iqs=iqs,
    )


class TestLoadSuite:
    def test_loads_fixture(self):
        suite = load_suite(FIXTURE)
        assert suite.name == "Support regression suite"
        assert suite.fail_below_iqs == 0.70
        assert suite.fail_below_groundedness == 0.60
        assert len(suite.examples) == 3
        assert suite.examples[0].tags == ["billing"]
        assert suite.examples[2].expected_iqs_min == 0.70

    def test_missing_file_raises(self):
        with pytest.raises(OSError):
            load_suite("tests/quality/does-not-exist.yaml")

    def test_malformed_yaml_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("examples: [unterminated", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_suite(str(bad))

    def test_non_mapping_top_level_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_suite(str(bad))

    def test_example_missing_required_fields_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "name: Bad suite\nexamples:\n  - query: \"Q only\"\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing 'query' or 'response'"):
            load_suite(str(bad))


class TestRunSuite:
    def test_all_pass(self):
        suite = EvalSuite(
            name="s",
            fail_below_iqs=0.70,
            fail_below_groundedness=None,
            examples=[
                EvalExample(query="q1", response="r1"),
                EvalExample(query="q2", response="r2"),
            ],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.9)):
            result = run_suite(suite)
        assert result.passed_count == 2
        assert result.failed_count == 0
        assert result.avg_iqs == pytest.approx(0.9)
        assert all(r.gate_reason is None for r in result.results)

    def test_failing_example_reports_gate_reason(self):
        suite = EvalSuite(
            name="s",
            fail_below_iqs=0.70,
            fail_below_groundedness=None,
            examples=[EvalExample(query="q1", response="r1")],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.5)):
            result = run_suite(suite)
        assert result.failed_count == 1
        assert result.results[0].gate_reason is not None

    def test_per_example_threshold_overrides_suite_default(self):
        suite = EvalSuite(
            name="s",
            fail_below_iqs=0.50,
            fail_below_groundedness=None,
            examples=[EvalExample(query="q1", response="r1", expected_iqs_min=0.95)],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.80)):
            result = run_suite(suite)
        # Passes the suite default (0.50) but not the example's own floor (0.95).
        assert result.failed_count == 1

    def test_cli_fail_below_overrides_suite_default(self):
        suite = EvalSuite(
            name="s",
            fail_below_iqs=0.50,
            fail_below_groundedness=None,
            examples=[EvalExample(query="q1", response="r1")],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.80)):
            result = run_suite(suite, fail_below=0.95)
        assert result.failed_count == 1

    def test_groundedness_floor_enforced(self):
        suite = EvalSuite(
            name="s",
            fail_below_iqs=0.70,
            fail_below_groundedness=0.95,
            examples=[EvalExample(query="q1", response="r1")],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.80, groundedness=0.50)):
            result = run_suite(suite)
        assert result.failed_count == 1
        assert "groundedness" in result.results[0].gate_reason


class TestFormatReport:
    def test_report_includes_summary(self):
        suite = EvalSuite(
            name="My suite",
            fail_below_iqs=0.70,
            fail_below_groundedness=None,
            examples=[EvalExample(query="q1", response="r1")],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.9)):
            result = run_suite(suite)
        report = format_report(suite, result)
        assert "My suite" in report
        assert "Summary: 1/1 passed" in report

    def test_report_lists_failures(self):
        suite = EvalSuite(
            name="My suite",
            fail_below_iqs=0.70,
            fail_below_groundedness=None,
            examples=[EvalExample(query="q1", response="r1", tags=["billing"])],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.5)):
            result = run_suite(suite)
        report = format_report(suite, result)
        assert "FAIL #1 [billing]" in report
        assert "q1" in report
        assert "Summary: 0/1 passed" in report


class TestFormatJunitXml:
    def test_all_passing_has_no_failures(self):
        suite = EvalSuite(
            name="My suite",
            fail_below_iqs=0.70,
            fail_below_groundedness=None,
            examples=[
                EvalExample(query="q1", response="r1"),
                EvalExample(query="q2", response="r2"),
            ],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.9)):
            result = run_suite(suite)
        xml = format_junit_xml(suite, result)

        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        assert root.tag == "testsuite"
        assert root.attrib["name"] == "My suite"
        assert root.attrib["tests"] == "2"
        assert root.attrib["failures"] == "0"
        testcases = root.findall("testcase")
        assert len(testcases) == 2
        assert all(tc.find("failure") is None for tc in testcases)

    def test_failing_example_has_failure_element(self):
        suite = EvalSuite(
            name="My suite",
            fail_below_iqs=0.70,
            fail_below_groundedness=None,
            examples=[EvalExample(query="q1", response="r1", tags=["billing"])],
        )
        with patch("scroot.cli.eval.score", return_value=_result(0.5)):
            result = run_suite(suite)
        xml = format_junit_xml(suite, result)

        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        assert root.attrib["tests"] == "1"
        assert root.attrib["failures"] == "1"
        testcase = root.find("testcase")
        failure = testcase.find("failure")
        assert failure is not None
        assert failure.attrib["message"]
        assert "q1" in failure.text


class TestEvalCommand:
    def test_passing_suite_exits_zero(self):
        with patch("scroot.cli.eval.score", return_value=_result(0.9)):
            result = runner.invoke(app, ["eval", "--suite", FIXTURE])
        assert result.exit_code == 0
        assert "Summary: 3/3 passed" in result.output

    def test_failing_suite_exits_nonzero(self):
        with patch("scroot.cli.eval.score", return_value=_result(0.1)):
            result = runner.invoke(app, ["eval", "--suite", FIXTURE])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_json_output(self):
        with patch("scroot.cli.eval.score", return_value=_result(0.9)):
            result = runner.invoke(app, ["eval", "--suite", FIXTURE, "--json"])
        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert data["passed"] == 3
        assert data["failed"] == 0
        assert len(data["results"]) == 3

    def test_missing_suite_file_exits_nonzero(self):
        result = runner.invoke(app, ["eval", "--suite", "tests/quality/nope.yaml"])
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_fail_below_flag_overrides(self):
        # IQS 0.80 passes the suite's default 0.70 but fails a stricter --fail-below.
        with patch("scroot.cli.eval.score", return_value=_result(0.80)):
            result = runner.invoke(
                app, ["eval", "--suite", FIXTURE, "--fail-below", "0.95"]
            )
        assert result.exit_code == 1

    def test_output_writes_junit_xml(self, tmp_path):
        output_path = tmp_path / "junit.xml"
        with patch("scroot.cli.eval.score", return_value=_result(0.9)):
            result = runner.invoke(
                app, ["eval", "--suite", FIXTURE, "--output", str(output_path)]
            )
        assert result.exit_code == 0
        assert output_path.exists()

        import xml.etree.ElementTree as ET

        root = ET.parse(output_path).getroot()
        assert root.tag == "testsuite"
        assert root.attrib["tests"] == "3"
        assert root.attrib["failures"] == "0"

    def test_output_junit_xml_with_failures(self, tmp_path):
        output_path = tmp_path / "junit.xml"
        with patch("scroot.cli.eval.score", return_value=_result(0.1)):
            result = runner.invoke(
                app, ["eval", "--suite", FIXTURE, "--output", str(output_path)]
            )
        assert result.exit_code == 1
        assert output_path.exists()

        import xml.etree.ElementTree as ET

        root = ET.parse(output_path).getroot()
        assert root.attrib["tests"] == "3"
        assert root.attrib["failures"] == "3"
        assert root.find("testcase/failure") is not None
