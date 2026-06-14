# Contributing to scroot

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/sunnyguntuka/scroot
cd scroot
pip install -e ".[dev,security,database]"
```

Optional: enable improved sentence splitting once:

```bash
python -c "import scroot; scroot.setup_nltk()"
```

## Running tests

```bash
pytest tests/ -v
```

The full test suite requires sentence-transformers and PyTorch. Model weights
are downloaded on first run (~500 MB). Subsequent runs use the local cache.

```bash
# Faster: skip slow NLI tests
pytest tests/ -v -k "not groundedness and not consistency and not core"
```

## Linting

```bash
ruff check src/ tests/
```

All PRs must pass `ruff` with zero errors.

## Code style

- No comments unless the *why* is non-obvious.
- No docstring bloat - one-line docstrings for obvious methods.
- All public functions need typed Args/Returns in their docstring.
- Default parameter values must be consistent between `Auditor.__init__`
  and the underlying metric functions.

## Adding a new metric

1. Create `src/scroot/metrics/my_metric.py` with a `score_my_metric(response, ...) -> tuple[float, dict]` function.
2. Add it to `Auditor.score()` in `core.py`.
3. Add its weight to `DEFAULT_WEIGHTS` in `composite.py`.
4. Add a flag condition to `detect_flags()` in `flags.py` if appropriate.
5. Write tests in `tests/test_my_metric.py`.

## Security

Before submitting any PR that touches `feedback/store.py`, `feedback/injector.py`,
or `models.py`, re-read the security notes in `CHANGELOG.md` to ensure
you haven't inadvertently reintroduced a patched vulnerability.

Key invariants to preserve:
- Model names must be validated against the allowlist before loading.
- All feedback fields must pass through `sanitize_for_prompt()` before
  prompt interpolation.
- `FeedbackStore` path must be validated against traversal before use.

## Pull request process

1. Fork the repo and create a feature branch.
2. Write tests first (TDD preferred).
3. Ensure `pytest tests/ -v` passes and `ruff check` is clean.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a PR with a clear description of the change and motivation.

## License

By contributing, you agree your contributions will be licensed under Apache-2.0.
