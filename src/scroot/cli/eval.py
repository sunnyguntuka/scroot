"""`scroot eval` - run a YAML-defined quality regression suite.

Loads a suite of (query, response, context) examples with expected IQS/
groundedness floors, scores each with scroot, and reports pass/fail using
EntailmentResult.passes_gate() / gate_reason(). Intended as a CI/CD quality
gate - exits non-zero if any example fails its gate.
"""

from __future__ import annotations

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


def run_suite(suite: EvalSuite, fail_below: "float | None" = None) -> EvalRunResult:
    """Score every example in a suite and evaluate its quality gate.

    Args:
        suite: The eval suite to run.
        fail_below: Optional CLI override for the IQS gate threshold,
            applied to examples that don't set their own
            ``expected_iqs_min``.

    Returns:
        EvalRunResult with per-example outcomes and aggregate stats.
    """
    results = []
    for example in suite.examples:
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
        results.append(ExampleResult(
            example=example,
            iqs=result.iqs,
            passed=reason is None,
            gate_reason=reason,
        ))

    return EvalRunResult(results=results)


def format_report(suite: EvalSuite, run_result: EvalRunResult) -> str:
    """Format a plain-text report of an eval run for CLI output."""
    lines = [f"Eval suite: {suite.name}", ""]

    for i, result in enumerate(run_result.results, start=1):
        if result.passed:
            continue
        tags = f" [{', '.join(result.example.tags)}]" if result.example.tags else ""
        lines.append(f"FAIL #{i}{tags}")
        lines.append(f"  Query: {result.example.query}")
        lines.append(f"  IQS:   {result.iqs:.2f}")
        lines.append(f"  Reason: {result.gate_reason}")
        lines.append("")

    lines.append(
        f"Summary: {run_result.passed_count}/{len(run_result.results)} passed "
        f"- avg IQS {run_result.avg_iqs:.2f}"
    )
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
