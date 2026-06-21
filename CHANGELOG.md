# Changelog

All notable changes to `scroot` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.4.0] - 2026-06-21

### Breaking changes

- **Default groundedness backbone changed** from `deberta-v3-base` to
  `MiniCheck-RoBERTa-Large`. Use `Auditor(groundedness_backbone="deberta-base")`
  to restore the previous behaviour. MiniCheck achieves AUC 0.991 vs 0.875 for
  deberta on NQ-500 hallucination discrimination; 1.75× mean latency cost.
- **Applicability gating is now ON by default** (`gate_inapplicable_dimensions=True`).
  Dimensions that are structurally inapplicable to a task (e.g. relevance on a
  generic "Summarize…" query) are excluded from IQS rather than collapsing the
  harmonic mean. Pass `Auditor(gate_inapplicable_dimensions=False)` to restore
  the previous behaviour.
- **Top-k premise pre-filtering is now ON by default** (`top_k_premises=5`). Pre-ranks
  NLI premises by embedding similarity and keeps the top 5 per claim, giving ~3.5×
  speedup on long-document contexts with 0.000 score delta. Pass
  `Auditor(top_k_premises=None)` to disable.

### New features

- **Selectable groundedness backbone**: `Auditor(groundedness_backbone=...)` accepts
  `"minicheck-roberta-large"` (default) or `"deberta-base"` (fast). Factory function
  `get_groundedness_backbone(name, device)` in `scroot.models`.
- **Applicability gating**: `Auditor(gate_inapplicable_dimensions=True)` detects and
  excludes structurally inapplicable IQS dimensions. `result.inapplicable_dimensions`
  lists gated dimensions for transparency.
- **Top-k premise pre-filtering**: `Auditor(top_k_premises=k)` pre-ranks NLI premises
  by embedding similarity before the cross-encoder runs. Lossless on NQ-500 (0.000 MAD
  vs unfiltered). See `benchmarks/results/topk_optimization.md`.

### Bug fixes

- **Groundedness: sentence-split context before NLI inference.** The NLI cross-encoder
  degrades to near-zero entailment when the premise is a full paragraph (known limitation).
  Fixed by sentence-splitting each context chunk before building NLI pairs. Effect:
  A0 mean groundedness 0.461 → 0.983.
- **Confidence: "May" month no longer matches as a hedge word.** The `\bmay\b` hedge
  pattern matched "May 18" (a date), scoring confident factual dates as hedged.
  Fixed with a negative lookahead: `\bmay\b(?!\s*\d)`.

### Benchmarks (v0.4.0 default)

- **Hallucination discrimination**: AUC 0.991 (MiniCheck), near-perfect, deterministic, $0.
- **Human correlation**: Spearman ρ 0.47 on SummEval (same 396 samples as RAGAS and
  DeepEval). Beats DeepEval (ρ 0.28); RAGAS leads at ρ 0.64 (LLM judge, non-deterministic).
- **Determinism**: 5,400+ checks, 0 deviations, end-to-end through `Auditor.score()`.
- Full methodology and reproducibility: see [BENCHMARKS.md](BENCHMARKS.md).

## [0.2.0] - 2026-06-12

First release since 0.1.0. Bundles the Context Builder, Review Console
hardening, Evidence Map, `scroot eval`, the IQS groundedness-exclusion
work, and numerous fixes. (0.1.1 and 0.1.2 were prepared but never published
to PyPI; their changes are included here.)

### Changed
- **IQS default formula is now the weighted harmonic mean** (`iqs_mode="harmonic"`),
  matching the formula documented in the README. Previously `Auditor` and
  `compute_iqs()` defaulted to `iqs_mode="geometric"`, which contradicted
  the README's "weighted harmonic mean" description. Pass
  `iqs_mode="geometric"` to `Auditor()` to keep the previous behaviour.
- `AgentRegistry.score()` now recomputes IQS using the underlying
  `Auditor`'s configured `iqs_mode` instead of always using the
  `compute_iqs()` default.
- Added `scroot score` CLI command for one-shot scoring from the command
  line (`scroot score --query ... --response ... --context ...`).

### Added
- `EntailmentResult.passes_gate()` and `EntailmentResult.gate_reason()`:
  inline quality gating with optional per-metric floors (e.g.
  `result.passes_gate(0.80, require_groundedness=0.95)`), with
  `gate_reason()` explaining why a gate failed.
- `EntailmentResult.iqs_explanation()`: deterministic, no-LLM one-sentence
  explanation of the IQS score, naming the weakest metric when below
  threshold.
- `EntailmentResult.weakest_metric` and `EntailmentResult.score_variance`
  properties for identifying which metric drove a low IQS and how spread
  out the five metric scores are.
- `ModelDownloadError`: raised with actionable, offline-pre-download
  instructions when a scoring model fails to download or load.
- `ContextBuilder`: request-scoped context accumulator for multi-step
  RAG and agentic pipelines. Carries grounding documents through the
  pipeline to `auditor.score()` without requiring code restructuring.
  Accepts LangChain `Document`, ChromaDB `QueryResult`, LlamaIndex
  nodes, Pinecone `ScoredVector`, dicts with a `text` key, and plain
  strings.
- PII scrubbing built into `ContextBuilder` by default: emails, phones,
  SSNs, credit cards, IP addresses, dates of birth, street addresses,
  names, and API keys/secrets replaced with typed placeholders before
  processing. Original values never stored.
- `ContextPayload` dataclass returned by `ContextBuilder.build()`,
  carrying assembled text, source provenance, token count, truncation
  flag, PII scrub summary, and SHA-256 checksum.
- `auditor.score()` now accepts `ContextPayload` and plain `str` in
  addition to `list[str]` and `None` for the `context` parameter.
- Audit logging for all context operations - SOC II compliant,
  content-free (counts, sources, checksums only).
- `scroot.configure_audit_log()` for configuring audit log destination
  (stderr / file / disabled) and retention.
- `session_id` and `context_checksum` fields on `CorrectionRecord` and
  the dashboard queue API for context audit-trail reconstruction.
- API reference at `docs/context_builder.md` and runnable demo notebook
  at `examples/context_builder.ipynb`.
- `EntailmentResult.metric_explanations`: per-metric, one-sentence
  explanations for any flagged metric (e.g. `groundedness` ->
  "The response makes claims that are not supported by the provided
  context."). Included in `to_dict()` and surfaced in the Review
  Console as hover tooltips on flagged metric bars, alongside the
  existing `iqs_explanation` and a "high spread" badge when
  `score_variance > 0.30`.
- Guardrail usage tracking ("loop closed"): `CorrectionRecord.guardrail_applied_count`
  is incremented whenever `GuardrailInjector.build_context()` includes
  that correction in a generated prompt. New `GET /api/guardrails/stats`
  endpoint and Review Console UI (sidebar stat + per-record "Guardrail
  status" line on Record Detail).
- `scroot eval` CLI: run a YAML-defined regression suite of
  (query, response, context) examples against `passes_gate()` /
  `gate_reason()` as a CI/CD quality gate. Supports `--fail-below` and
  `--json`, exits non-zero on any failing example. Requires `pyyaml`
  (now part of the `dashboard` and `ci` extras).
- `scroot eval --output junit.xml`: write a JUnit XML report alongside
  the normal text/JSON output, for CI systems (GitHub Actions, GitLab,
  Jenkins) that render per-example pass/fail in the test results UI.
- Evidence Map: `EntailmentResult.evidence_map` (an `EvidenceMap` of
  `EvidenceEntry` items, both new exported dataclasses, built by
  `build_evidence_map()`) gives sentence-level NLI attribution - for each
  response sentence, the best-matching context chunk and whether it is
  supported, contradicted, or has no grounding. Computed automatically
  whenever `context` is provided (disable with
  `Auditor(compute_evidence_map=False)`); included in `to_dict()` and
  surfaced in the Review Console as a new "Evidence map" panel on Record
  Detail with sentence-level color coding and a coverage summary.

### Changed
- `auditor.score()` `context` parameter now typed as
  `ContextPayload | str | list[str] | None` (was `list[str] | None`).
  Fully backward compatible - existing `list[str]` usage unchanged.

### Changed
- **IQS now formally excludes groundedness when no context is provided.**
  `groundedness` is `None` (inapplicable), not `0.0`, and is dropped from the
  IQS computation with its weight redistributed proportionally across the
  remaining four metrics - so a no-context response is scored on 4 metrics, not
  penalised as if groundedness were zero. A genuine `0.0` groundedness (context
  provided, NLI found no support) is still included and correctly drives IQS to
  0. New `compute_iqs_detailed(scores, weights, mode)` returns
  `(iqs, effective_weights)`; the existing positional `compute_iqs(...)` is
  unchanged and now delegates to it.
- `EntailmentResult.passes_gate()` / `gate_reason()` now **fail open** on a
  per-metric floor for a metric that was not scored (e.g.
  `require_groundedness` with no context): the floor is skipped with a
  `GroundednessUnavailableWarning` instead of failing the gate. The IQS
  threshold still applies. (Previously this failed the gate.)

### Added
- `EntailmentResult.effective_weights`, `.context_used`, and
  `.iqs_metric_count`: which metrics contributed to IQS and at what
  (redistributed) weight. Included in `to_dict()` and surfaced in the Review
  Console - the groundedness bar shows a neutral "— not scored" state and the
  IQS shows a `(4/5)` indicator when scored without context.
- `NoContextWarning` (emitted by `auditor.score()` when called without
  context, unless the groundedness weight is 0), `GroundednessUnavailableWarning`
  (a groundedness floor was requested but groundedness is `None`), and
  `GroundednessComputationError` (context provided but groundedness scoring
  raised - degrades to `None` instead of failing the call). All exported from
  the top-level package.
- `auditor.score()` now treats empty/whitespace-only context (`""`, `"   "`,
  `[]`) identically to `None` - groundedness is not computed.

### Fixed
- Dashboard SPA deep links / refreshes (`/queue`, `/analytics`, `/queue/{id}`)
  returned 404 in a real deployment: the server served the built UI at `/` but
  had no history-API fallback. It now serves `index.html` for non-API routes
  (BrowserRouter fallback) while unknown `/api/*` paths still return 404.
- `/api/queue` returned HTTP 500 whenever any record had status `applied`: the
  `QueueItem` response model's `status` Literal omitted `applied` (a documented
  `CorrectionRecord` state). Added it.

### Security
- Review Console hardening (pre-release audit fixes):
  - `GET /api/settings` no longer returns the stored LLM provider API key.
    It now exposes only a masked hint (`sk-a…wxyz`) and an `api_key_set`
    boolean; the key is write-only (a blank submitted key leaves the stored
    one unchanged).
  - Config files holding the provider key (`~/.scroot/config.json` and the
    working-dir `.scroot_settings.json`) are now written with `0600`
    (owner-only) permissions.
  - API-based correction validates the configured `base_url` against an
    allowlist of known provider hosts (plus local Ollama), blocking key
    exfiltration to untrusted hosts and SSRF to internal/metadata endpoints.
    Override for a trusted custom gateway with `SCROOT_ALLOW_ANY_BASE_URL=1`.
  - Optional dashboard token auth: set `SCROOT_DASHBOARD_TOKEN` or pass
    `scroot serve --token`. When set, all `/api/*` routes (except
    `/api/health`) require the token. Binding to a non-loopback host without
    a token now emits a startup warning.
- Packaging: the built Review Console UI (`ui/dist`) is now included in the
  wheel, so `scroot serve` works after `pip install scroot`.
- `CorrectorConfig.load()` now warns (instead of silently resetting to
  defaults) when `~/.scroot/config.json` is unreadable or corrupt, so a bad
  file is not silently overwritten - losing the stored key/settings - on the
  next save. Several broad `except Exception` handlers narrowed to the
  expected error types.
- Documented the dashboard trust model and the automated dependency-audit
  posture (CI `pip-audit`/`npm audit`/license check) in `docs/security.md`.
- Input size limits on `ContextBuilder`: 50,000 chars per chunk,
  500 chunks per call, configurable `max_tokens` total budget,
  128-char session IDs, bounded metadata dicts.
- `ContextSealedError` raised on mutation after `build()` - creates a
  clear audit boundary and prevents concurrent-mutation races.
- `[SECRET]` PII entity type detects and redacts API keys, tokens,
  and long hex strings from context content.
- `SecurityWarning` emitted when `pii_scrub=False` is used with
  `SCROOT_ENV=production`.

---

## [0.1.2] - 2026-06-07

### Added
- **Review Console** - local web dashboard (`scroot serve`, port 7432).
  Built with React 18 + Vite, FastAPI backend, Tailwind CSS v3 "Soft Clinical"
  design system (indigo palette, DM Sans + JetBrains Mono typography).
  Five pages: Inbox (review queue), Record Detail, Analytics, Pipeline,
  Export. No cloud dependency. Runs fully offline.
- **IQS health pill** on every dashboard page - ambient avg IQS signal
  with pass / warn / fail colour coding.
- **Correction pipeline** - batch-correct flagged records using any
  OpenAI-compatible LLM. NLI re-scores every draft before committing;
  corrections below improvement threshold return to review queue.
  Three modes: Generate drafts only, Auto-commit if NLI passes,
  Fully autonomous (Scroot Cloud).
- **Export page** - download reviewed corrections as JSONL or CSV with
  filters for status, agent, date range, and minimum IQS improvement.
  Fine-tuning readiness indicator (progress toward 50 corrected records).
- **Analytics page** - 30-day IQS trend (LineChart), flag frequency
  (BarChart per metric), score distribution histogram, per-agent
  breakdown table sorted by avg IQS ascending.
- **Demo mode** - append `?demo` to the URL to populate all pages with
  realistic sample data without a running backend.
- `feedback.store.FeedbackStore`: `get_pending()`, `mark_reviewed()`,
  `export_for_finetuning()` methods; `status` field on `CorrectionRecord`.

### Changed
- IQS harmonic mean now uses semantic retrieval for completeness and
  bidirectional consistency scoring.
- Default metric weights revised: groundedness 0.35, completeness 0.25,
  relevance 0.20, consistency 0.15, confidence 0.05.

---

## [0.1.1] - 2026-06-03

### Added
- Geometric mean IQS mode (`mode="geometric"`) alongside harmonic mean.
- Atomic claim extraction for groundedness scoring.
- Bi-encoder fallback when cross-encoder score is unavailable.
- `RAG_WEIGHTS` preset: groundedness-heavy weight profile for RAG pipelines.
- 60+ confidence vocabulary patterns covering uncertainty, hedging,
  and assertion strength.
- Confidence accuracy, completeness accuracy, and paraphrase groundedness
  benchmark suites.

### Changed
- IQS correlation with human judgements improved from 0.37 → 0.69
  (500-example NQ benchmark). Beats DeepEval (0.71 on their test set),
  RAGAS (0.68), and TruthScore (0.63) on a shared 500-example evaluation.
- `score_consistency` now uses larger DeBERTa-v3-large NLI model when
  available (falls back to base).

### Fixed
- Python 3.9 compatibility (`from __future__ import annotations` in
  `groundedness.py`).
- `test_composite` mode argument passing for harmonic/geometric tests.

---

## [0.1.0] - 2026-06-02

Initial public release.

### Added
- `Auditor` class: orchestrates all five metrics, returns `EntailmentResult`.
- Metrics: `score_groundedness`, `score_completeness`, `score_relevance`,
  `score_consistency`, `score_confidence`.
- `compute_iqs`: weighted harmonic mean composite score (IQS).
- `detect_flags`: flags quality issues from metric scores
  (`hallucination_risk`, `off_topic`, `self_contradictory`, `incomplete`,
  `ungrounded`).
- `EntailmentResult` dataclass with `.to_dict()` and `.__repr__()`.
- Top-level convenience functions: `scroot.score()`, `scroot.verify()`.
- `AgentRegistry`: per-agent routing layer with custom IQS weights,
  thresholds, context requirements, metadata, and per-agent statistics.
  Thread-safe. Duck-type compatible with `Auditor` for `sample_and_score`
  and `DatabaseConnector`.
- `sample_and_score`: five sampling strategies (`random`, `percentage`,
  `stratified`, `confidence`, `priority`) via `SamplingResult` with
  aggregate statistics, per-stratum breakdowns, and 95% confidence
  interval for the population mean IQS.
- `DatabaseConnector`: score LLM responses stored in any
  SQLAlchemy-compatible database (PostgreSQL, MySQL, SQLite, BigQuery,
  Snowflake). Methods: `score_all`, `score_where`, `score_sampled`,
  `score_incremental`, `fetch`, `write_result`. Optional dependency:
  `pip install "scroot[database]"`.
- `FeedbackStore`: append-only JSONL correction store with optional
  Fernet encryption at rest, field masking, `delete()`, `purge()`,
  `max_records`, `ttl_days`, and `validate_integrity()`.
- `GuardrailInjector`: builds correction context for LLM system prompts.
  Three strategies: `recent`, `relevant`, `rules`. PII scrubbing
  (SSN, email, phone, credit card) and prompt-injection sanitization
  applied before interpolation.
- `CorrectionRecord` dataclass with sequential `record_number` and
  SHA-256 `record_hash` for tamper detection.
- Lazy model loading with per-key singleton cache, bounded LRU
  (max 10 entries), thread-safe.
- Model allowlist (`DEFAULT_ALLOWED_MODELS`) with `trust_model()` for
  authorizing custom or fine-tuned models.
- Configurable thresholds: `entailment_threshold`, `coverage_threshold`,
  `contradiction_threshold`, `max_sentences`.
- Input length limits on all `Auditor` inputs for DoS protection:
  `max_query_length`, `max_response_length`, `max_context_items`,
  `max_context_item_length`, `max_batch_size`.
- Path traversal validation on `FeedbackStore` path.
- Optional NLTK sentence splitting via `scroot.setup_nltk()`.

### Performance
- `import scroot` < 250ms (sentence-transformers imported lazily).
- `score_groundedness` batches all (chunk, claim) pairs into a single
  `model.predict()` call.
- `score_consistency` caps at `max_sentences=25` (first/last half) to
  bound O(n²) pairwise NLI cost.
