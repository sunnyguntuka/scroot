# Contributing to scroot

Thanks for your interest in contributing! This guide covers local setup,
testing, and the conventions we use.

## Setup

```bash
git clone https://github.com/sunnyguntuka/scroot.git
cd scroot
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -e ".[dev]"
```

Optional extras for working on specific areas:

```bash
pip install -e ".[dashboard]"   # dashboard backend (FastAPI)
pip install -e ".[local]"       # local LLM corrector (llama-cpp-python)
pip install -e ".[api]"         # API-based corrector (OpenAI)
pip install -e ".[bench]"       # benchmark suite
```

For the dashboard frontend:

```bash
cd src/scroot/ui
npm install
npm run dev
```

## Running checks

```bash
ruff check src/ tests/

# Fast tests (skip tests that need model weights)
pytest -m "not needs_model" --cov=src/scroot

# Full suite, including tests that download/run model weights
pytest
```

## Pre-commit hooks (recommended)

This repo uses [`pre-commit`](https://pre-commit.com/) with
[`detect-secrets`](https://github.com/Yelp/detect-secrets) to catch
accidental credential commits before they're pushed:

```bash
pip install pre-commit
pre-commit install
```

If `detect-secrets` flags a new finding that's a false positive (e.g. a
test fixture), update the baseline:

```bash
pip install detect-secrets
detect-secrets scan --baseline .secrets.baseline
```

## Coding conventions

- Code is formatted/linted with [ruff](https://docs.astral.sh/ruff/)
  (`line-length = 88`, target Python 3.9+).
- New code in `src/scroot/corrector/` must maintain the project's
  95% coverage gate - it enforces the NLI re-score invariant (every
  correction is re-scored before being accepted).
- `scroot[local]` (`llama-cpp-python`) must never be a dependency of
  the base install - it's strictly opt-in.

## Pull requests

- Keep PRs focused on a single change.
- Add or update tests for any behavior change.
- Update `README.md` / `docs/` / `CHANGELOG.md` if the change is
  user-facing.
- CI runs lint, tests (Python 3.9/3.11/3.12), a wheel smoke test,
  dependency/license/secrets audits, and a PyPI install smoke test.
  All must pass before merge.

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities - see
[`docs/security.md`](../docs/security.md) for the responsible disclosure
process.
