# Security

## Responsible disclosure

If you discover a security vulnerability in scroot, please report it
privately by emailing **guntuka.sunny@gmail.com** rather than opening a
public GitHub issue. Include a description of the issue, steps to
reproduce, and the affected version. We aim to acknowledge reports within
a few business days.

## No telemetry

scroot does not collect, transmit, or report any usage data, metrics, or
analytics. Nothing about your queries, responses, scores, or configuration
is sent anywhere by the library itself.

## Data residency - what leaves the machine

By default, **nothing leaves the machine**:

- `auditor.score()` runs the NLI cross-encoder and embedding model
  locally (CPU or GPU). Query, response, and context text never leave
  the process.
- `ContextBuilder` assembles and scrubs context in-memory; nothing is
  written to disk unless an `encryption_key` is supplied (Fernet
  encryption at rest - see below).
- `scroot[local]` runs the corrector LLM locally via
  `llama-cpp-python`. The `[local]` extra and the corresponding model
  weights are strictly opt-in: a base `pip install scroot` never
  downloads or runs an LLM.
- `scroot[api]` is the only path that sends data to a third-party
  API, and only when you explicitly configure an API-based corrector
  with your own API key.

## ContextBuilder PII scrubbing

`ContextBuilder` scrubs personally identifiable information from
retrieved context **by default** (`pii_scrub=True`). Before context is
assembled or scored, the following entity types are detected and
replaced with typed placeholders (e.g. `[EMAIL]`, `[PHONE]`):

- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses
- Dates of birth
- Street addresses
- Names
- **`[SECRET]`** - API keys, tokens, and credentials embedded in
  retrieved documents (e.g. a leaked key pasted into a knowledge-base
  article). This prevents scroot from accidentally logging or
  surfacing credentials that appear in retrieved content.

Original values are never stored - only the scrub summary (counts per
entity type) is retained for the audit trail. See
[docs/context_builder.md](context_builder.md) for details and how to
disable scrubbing (`pii_scrub=False`) if your pipeline already handles
PII upstream.

## Encryption at rest

When `ContextBuilder(encryption_key=...)` is provided, context payloads
persisted to disk are encrypted using
[Fernet](https://cryptography.io/en/latest/fernet/) symmetric
encryption (via the `cryptography` package, `pip install
'scroot[security]'`). Without an `encryption_key`, `ContextBuilder`
does not write context to disk at all.

## Audit logging

All `ContextBuilder` operations emit structured, content-free audit
events (counts, sources, checksums - never the underlying text), via
`scroot.configure_audit_log()`. Default destination is stderr; for SOC
II environments, route to a JSONL file with retention-based rotation.

## Known limitation: database connector and SQL injection

`scroot.connectors.DatabaseConnector` (`pip install
'scroot[database]'`) builds SQL statements using string interpolation
for **table names, column names, and `WHERE`/cursor-column clauses**.
SQLAlchemy bind parameters protect *values* (e.g. row data written via
`write_result()`), but table/column identifiers cannot be parameterised
the same way.

**Risk:** if `source_table`, `result_table`, `column_map` values, or the
`where` / `cursor_column` arguments are derived from untrusted input,
this is a SQL injection vector.

**Current mitigation:** constructing a `DatabaseConnector` emits a
`scroot.connectors.SecurityWarning`. Only pass identifiers and WHERE
clauses that you control (e.g. from your own config files), never from
end-user input.

**Planned hardening (tracked for v0.3.0):**

- Allowlist validation for table names and column names
- Parameterise all filter values via `cursor.execute(sql, params)`
- A `dry_run=True` mode that returns the generated SQL without executing it
- Schema validation before any write operation
- Migration script support for schema changes

## Review Console (dashboard) trust model

The `scroot serve` dashboard is a **local, single-user, unauthenticated**
tool. Its security posture depends on how it is bound:

- **Default (`127.0.0.1`)** - single-user safe. Only processes on the local
  machine can reach the API.
- **Non-loopback bind (e.g. `--host 0.0.0.0`)** - exposes the entire
  correction store *and* the stored LLM provider key to the network. scroot
  emits a startup warning in this case. **Always** set a shared token and/or
  run behind an authenticating reverse proxy.

Controls in place:

- **No key disclosure (H-1):** `GET /api/settings` never returns the raw
  provider API key - only a masked hint (`sk-a…wxyz`) and an `api_key_set`
  boolean. The key is write-only: submitting a blank key leaves the stored
  one unchanged.
- **File permissions (M-1):** `~/.scroot/config.json` and the working-dir
  `.scroot_settings.json` are written with `0600` (owner-only).
- **Outbound endpoint allowlist (M-2):** API-based correction will only send
  the key to known provider hosts (`api.openai.com`, `api.anthropic.com`,
  `api.groq.com`, `openrouter.ai`, Google Gemini) or a local Ollama endpoint.
  This blocks pointing the server at an attacker host (key theft) or an
  internal/metadata address (SSRF). Override for a trusted custom gateway with
  `SCROOT_ALLOW_ANY_BASE_URL=1`.
- **Optional token auth (H-2):** set `SCROOT_DASHBOARD_TOKEN` (or pass
  `--token`). When set, every `/api/*` route except `/api/health` requires the
  token via `Authorization: Bearer <token>` or `X-Scroot-Token`.

The feedback store itself is plaintext JSONL by default. For sensitive data,
construct `FeedbackStore` with an `encryption_key` (Fernet, at-rest
encryption) and/or a `field_mask` for fields like `query`/`context_used`.

## scroot-cloud license enforcement — residual risks

scroot-cloud uses offline Ed25519 signature verification with a last-seen
timestamp guard (stored in `~/.scroot/state`). This section documents what the
technical controls prevent and what they do not, so buyers and operators can make
informed risk decisions.

### What Ed25519 prevents

- **Forged tokens**: only the holder of the Ed25519 private key (stored in KMS /
  Vault outside both repos) can mint a valid token. Any bit flip in the payload
  causes `BadSignatureError`.
- **Tampered claims**: changing tier, features, or expiry without re-signing
  causes `BadSignatureError`.
- **Expired tokens**: `verify_license()` rejects tokens past their `expires`
  timestamp.
- **Naive clock rollback**: `_check_clock_rollback()` raises `LicenseError` if
  the current time is more than 30 seconds before the last-seen timestamp,
  mitigating the most common attack (rolling back the system clock to reuse an
  expired token).

### What the technical controls do NOT prevent

| Scenario | Why not prevented | Mitigation |
|---|---|---|
| Token replay across machines | Offline tokens are stateless by design (air-gapped market requirement) | Customer agreement (contractual) |
| Patching out `get_enterprise()` calls | scroot is Apache-2.0 — forks may remove the call | The cloud implementation is absent; patching the call gains no functionality |
| Deleting `~/.scroot/state` | An operator with local write access controls the state file | Short token expiry (≤90 days) + auto-renew |
| Time-of-check / time-of-use within a process | `verify_license()` is called once at plugin load; the `License` object is held in-memory | Runtime checks use the already-verified object; no network-accessible path to force re-verification |

**On-premises and air-gapped enforcement is legal and commercial, not absolute
technical.** The signature guarantees authenticity of the token; it does not
prevent a licensee with local system access from keeping an expired token in use
by deleting state, rolling back system time, or patching source. These are
contractual violations, not security vulnerabilities. For high-assurance
environments, combine short token TTLs with automated renewal and host-level
integrity monitoring.

---

## Dependency supply chain

- All dependencies are pinned to compatible version ranges in
  `pyproject.toml`.
- scroot is licensed Apache-2.0; transitive dependencies are audited
  for license compatibility (MIT, Apache-2.0, BSD, LGPL).
- Run `pip-audit` against your installed environment to check for known
  vulnerabilities in dependencies.

### Automated vulnerability monitoring

The `dependency-audit` job in `.github/workflows/tests.yml` runs on every push
and PR and performs three checks:

- `pip-audit` over the Python environment,
- `npm audit --omit=dev` over the dashboard UI dependencies,
- `pip-licenses` with a hard fail on GPL/AGPL/LGPL copyleft licenses.

The `pip-audit` step is intentionally **non-blocking** (`continue-on-error`).
scroot's heaviest dependencies - `torch` and `transformers`, pulled in
transitively via `sentence-transformers` - routinely carry CVEs that lag behind
upstream fixes (e.g. as of this writing `transformers` CVE-2026-1839 is only
resolved in a `5.0.0rc3` pre-release). Forcing a blocking gate would break CI on
issues scroot cannot directly remediate, so instead the job **surfaces** them
and Dependabot opens upgrade PRs as stable fixes land.

**Guidance:** these CVEs are inference-time issues in the model runtime; they
require loading untrusted model weights or inputs. scroot's
`DEFAULT_ALLOWED_MODELS` allowlist (see *Model loading*) keeps you on vetted
models by default. Keep `torch`/`transformers` current in your own environment,
and once a stable `transformers` release carries the fix, add a minimum-version
floor in your deployment (scroot does not pin a pre-release floor itself).
