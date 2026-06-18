"""`scroot eval` - run a YAML-defined quality regression suite.

Loads a suite of (query, response, context) examples with expected IQS/
groundedness floors, scores each with scroot, and reports pass/fail using
EntailmentResult.passes_gate() / gate_reason(). Intended as a CI/CD quality
gate - exits non-zero if any example fails its gate.

Baseline comparison: pass ``--baseline last_run.json`` to compare the current
run against a prior run persisted with ``--save-baseline``. Per-case IQS
deltas are shown and ``--fail-on-regression`` exits non-zero when any case
regresses beyond the tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from scroot import score


@dataclass
class EvalExample:
    """A single (query, response, context) case in an eval suite."""

    query: str
    response: str
    context: "str | list[str] | None" = None
    expected_iqs_min: "float | None" = None
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalSuite:
    """A YAML-defined collection of EvalExamples with default gate thresholds."""

    name: str
    examples: list[EvalExample]
    fail_below_iqs: "float | None" = None
    fail_below_groundedness: "float | None" = None


@dataclass
class ExampleResult:
    """Outcome of scoring a single EvalExample."""

    example: EvalExample
    iqs: float
    passed: bool
    gate_reason: "str | None"
    baseline_iqs: "float | None" = None

    @property
    def iqs_delta(self) -> "float | None":
        """IQS change vs baseline (negative = regression)."""
        if self.baseline_iqs is None:
            return None
        return round(self.iqs - self.baseline_iqs, 4)


@dataclass
class EvalRunResult:
    """Aggregate outcome of running an EvalSuite."""

    results: list[ExampleResult]

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def avg_iqs(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.iqs for r in self.results) / len(self.results)

    def has_regression(self, tolerance: float = 0.02) -> bool:
        """Return True if any case's IQS dropped more than ``tolerance`` vs baseline."""
        for r in self.results:
            delta = r.iqs_delta
            if delta is not None and delta < -tolerance:
                return True
        return False

    def to_baseline(self) -> dict:
        """Serialise this run for use as a future baseline file."""
        return {
            "avg_iqs": round(self.avg_iqs, 4),
            "cases": [
                {
                    "query": r.example.query[:200],
                    "iqs": round(r.iqs, 4),
                }
                for r in self.results
            ],
        }

    def save_baseline(self, path: str) -> None:
        """Write this run's per-case scores to a JSON file for future comparison."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_baseline(), f, indent=2)


def load_baseline(path: str) -> "dict | None":
    """Load a baseline JSON file previously written by ``save_baseline``.

    Returns ``None`` if the file does not exist (first-run behaviour).
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "pyyaml is required for `scroot eval`. "
            "Install it with: pip install pyyaml"
        ) from exc
    return yaml


def load_suite(path: str) -> EvalSuite:
    """Load an EvalSuite from a YAML file.

    Expected shape:

        name: Support regression suite
        fail_below_iqs: 0.70
        fail_below_groundedness: 0.80
        examples:
          - query: "..."
            response: "..."
            context: "..."          # or a list of strings
            expected_iqs_min: 0.75  # optional, overrides fail_below_iqs
            tags: [billing]

    Args:
        path: Path to the YAML suite file.

    Returns:
        Parsed EvalSuite.

    Raises:
        RuntimeError: If pyyaml is not installed.
        OSError: If the file cannot be read.
        ValueError: If the YAML is malformed or missing required fields.
    """
    yaml = _import_yaml()

    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    raw_examples = data.get("examples") or []
    if not isinstance(raw_examples, list):
        raise ValueError(f"{path}: 'examples' must be a list")

    examples = []
    for i, raw in enumerate(raw_examples):
        if "query" not in raw or "response" not in raw:
            raise ValueError(f"{path}: examples[{i}] is missing 'query' or 'response'")
        examples.append(EvalExample(
            query=raw["query"],
            response=raw["response"],
            context=raw.get("context"),
            expected_iqs_min=raw.get("expected_iqs_min"),
            tags=raw.get("tags") or [],
        ))

    return EvalSuite(
        name=data.get("name", path),
        examples=examples,
        fail_below_iqs=data.get("fail_below_iqs"),
        fail_below_groundedness=data.get("fail_below_groundedness"),
    )


def run_suite(
    suite: EvalSuite,
    fail_below: "float | None" = None,
    baseline: "dict | None" = None,
) -> EvalRunResult:
    """Score every example in a suite and evaluate its quality gate.

    Args:
        suite: The eval suite to run.
        fail_below: Optional CLI override for the IQS gate threshold,
            applied to examples that don't set their own ``expected_iqs_min``.
        baseline: Optional baseline dict (from ``load_baseline()``) to compare
            against. When provided, each ``ExampleResult`` gets a
            ``baseline_iqs`` field showing the prior run's score.

    Returns:
        EvalRunResult with per-example outcomes and aggregate stats.
    """
    baseline_cases: list[dict] = (baseline or {}).get("cases", [])

    results = []
    for i, example in enumerate(suite.examples):
        result = score(query=example.query, response=example.response, context=example.context)
        threshold = (
            example.expected_iqs_min
            if example.expected_iqs_min is not None
            else fail_below
            if fail_below is not None
            else suite.fail_below_iqs
            if suite.fail_below_iqs is not None
            else 0.70
        )
        reason = result.gate_reason(
            threshold=threshold,
            require_groundedness=suite.fail_below_groundedness,
        )
        # Match baseline by index (order is stable within a suite)
        base_iqs: float | None = None
        if i < len(baseline_cases):
            base_iqs = baseline_cases[i].get("iqs")

        results.append(ExampleResult(
            example=example,
            iqs=result.iqs,
            passed=reason is None,
            gate_reason=reason,
            baseline_iqs=base_iqs,
        ))

    return EvalRunResult(results=results)


def format_report(suite: EvalSuite, run_result: EvalRunResult) -> str:
    """Format a plain-text report of an eval run for CLI output."""
    has_baseline = any(r.baseline_iqs is not None for r in run_result.results)
    lines = [f"Eval suite: {suite.name}", ""]

    for i, result in enumerate(run_result.results, start=1):
        delta = result.iqs_delta
        regression = delta is not None and delta < -0.02
        if result.passed and not regression:
            continue
        tags = f" [{', '.join(result.example.tags)}]" if result.example.tags else ""
        status = "FAIL" if not result.passed else "REGRESSED"
        lines.append(f"{status} #{i}{tags}")
        lines.append(f"  Query: {result.example.query}")
        if delta is not None:
            arrow = "↓" if delta < 0 else "↑"
            lines.append(f"  IQS:   {result.iqs:.2f}  ({arrow}{abs(delta):.3f} vs baseline {result.baseline_iqs:.2f})")
        else:
            lines.append(f"  IQS:   {result.iqs:.2f}")
        if result.gate_reason:
            lines.append(f"  Reason: {result.gate_reason}")
        lines.append("")

    summary = (
        f"Summary: {run_result.passed_count}/{len(run_result.results)} passed "
        f"- avg IQS {run_result.avg_iqs:.2f}"
    )
    if has_baseline:
        regressions = [r for r in run_result.results if r.iqs_delta is not None and r.iqs_delta < -0.02]
        summary += f" - {len(regressions)} regression(s) vs baseline"
    lines.append(summary)
    return "\n".join(lines)


def format_junit_xml(suite: EvalSuite, run_result: EvalRunResult) -> str:
    """Format an eval run as a JUnit XML report for CI integration.

    Each example becomes a ``<testcase>``; failing examples (per
    ``passes_gate()``/``gate_reason()``) get a ``<failure>`` child with the
    gate reason as the failure message.
    """
    testsuite = ET.Element("testsuite", {
        "name": suite.name,
        "tests": str(len(run_result.results)),
        "failures": str(run_result.failed_count),
    })

    for i, result in enumerate(run_result.results, start=1):
        tags = ", ".join(result.example.tags) if result.example.tags else ""
        case_name = f"#{i} {tags}".strip() if tags else f"#{i} {result.example.query[:40]}"
        testcase = ET.SubElement(testsuite, "testcase", {
            "classname": suite.name,
            "name": case_name,
        })
        if not result.passed:
            failure = ET.SubElement(testcase, "failure", {
                "message": result.gate_reason or "gate failed",
            })
            failure.text = (
                f"Query: {result.example.query}\n"
                f"IQS: {result.iqs:.2f}\n"
                f"Reason: {result.gate_reason}"
            )

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(testsuite, encoding="unicode")
