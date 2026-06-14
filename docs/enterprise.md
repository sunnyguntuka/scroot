# scroot for enterprises

scroot is **open core**: the full scoring engine, `ContextBuilder`,
local LLM corrector, CLI, and a single-user review dashboard are all
Apache-2.0 and free to use, including in commercial products. This page
explains what's in the open-source package today, what Scroot
Enterprise/Cloud adds on top, and how the boundary between the two is
enforced.

## What's in open source (free, forever)

- **`Auditor.score()` / `score_batch()`** - all five quality metrics
  (groundedness, completeness, relevance, consistency, confidence) and the
  IQS composite, fully local, no API key.
- **`ContextBuilder`** - RAG context assembly, PII scrubbing, and audit
  logging.
- **`scroot[local]`** - the Phi-4-mini / SmolLM3 local corrector for
  closing the feedback loop, with no API key required.
- **CLI** (`scroot score`, `scroot serve`, `scroot download-model`,
  etc.) and the **review dashboard** (`scroot serve`) for a single user
  / single SQLite or JSONL store.
- **CSV export** and **single-destination S3 export** of flagged records.
- Dependency, license, and secrets scanning are part of the project's own
  CI - see [`docs/security.md`](security.md) and
  [`docs/licenses.md`](licenses.md).

## What Scroot Enterprise/Cloud adds

| Capability | OSS | Enterprise/Cloud |
|---|---|---|
| Scoring engine, `ContextBuilder`, local corrector | ✅ | ✅ |
| Review dashboard | ✅ single user, local store | ✅ multi-tenant, hosted |
| Export destinations | ✅ CSV, single S3 bucket | ✅ multi-destination scheduling (S3, GCS, Snowflake, BigQuery, ...) |
| Authentication / SSO | - | ✅ SAML/OIDC SSO, RBAC for the review queue |
| Hosted deployment | - | ✅ managed dashboard at scroot.dev/cloud |
| Data residency | local-only by default (see [security.md](security.md)) | VPC and cloud-tier deployment options |
| Support | community (GitHub issues) | SLA-backed support |
| Audit log retention & compliance reporting | local JSONL/file, no retention policy | retention policies, compliance exports |

The OSS package will never require a license key, phone home, or degrade
in functionality over time. Enterprise features are additive - they live
behind separate, clearly-labeled entry points.

## How the boundary is enforced

Enterprise-only functionality is gated **at the API level**, not hidden in
the UI:

- `create_app(hosted=True)` (the hosted multi-tenant mode) raises
  `NotImplementedError` in the OSS package - see
  [`src/scroot/dashboard/server.py`](../src/scroot/dashboard/server.py).
- The `/export/push-s3` endpoint queues a single-destination job in OSS
  and returns a `note` pointing to Enterprise for multi-destination
  scheduling - see
  [`src/scroot/dashboard/routers/export.py`](../src/scroot/dashboard/routers/export.py).
- `--hosted` on the dashboard CLI is intentionally hidden
  (`argparse.SUPPRESS`) and unimplemented in OSS.

If you find a code path where an enterprise feature is hidden only in the
UI but reachable via the API, please file a bug - that's a boundary
violation per our design and will be fixed.

## Talk to us

For Enterprise/Cloud pricing, hosted deployments, or VPC/data-residency
requirements, visit https://scroot.dev/cloud or open a discussion on
[GitHub](https://github.com/sunnyguntuka/scroot/discussions).
