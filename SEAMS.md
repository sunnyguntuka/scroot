# SEAMS.md — scroot gated-surface audit

> **Auto-generated** by `scripts/list_seams.py`. Do not edit by hand.
> Re-generate: `python scripts/list_seams.py --write`

Every row below is a place where scroot delegates to scroot-cloud.
The OSS counterpart column shows what is available under Apache-2.0
with no scroot-cloud installed. The open algorithm, calibration engine,
local review UI, and air-gapped runtime are OSS. Only the operated,
governed, audited lifecycle is gated.

Learn more: https://scroot.dev/cloud

| Seam key | Label | OSS counterpart | Call site(s) |
|---|---|---|---|
| `audit.export` | Audit evidence export | result.evidence_map (scoring data, no signing/retention) | `src/scroot/audit.py:153` |
| `calibration.schedule` | Managed calibration lifecycle | calibrate() fit + CalibrationResult | `src/scroot/calibrate.py:216` |
| `drift.continuous` | Continuous drift monitoring (Ampulla) | regression_check() point-in-time CI gate | `src/scroot/drift.py:244` |
| `metrics.builder` | No-code custom metric builder | register_metric() code-level custom metric | `src/scroot/metrics/__init__.py:15` |
| `pii.policy` | Regulatory PII policy management | scrub() local masking with allowlist + grounding mode | `src/scroot/pii.py:44` |
| `review.queue` | Hosted review queue | review.ui() local single-user viewer (scroot serve) | `src/scroot/review/__init__.py:41` |
| `runtime.managed` | Managed scoring runtime | runtime.run() local air-gapped scoring + preflight() | `src/scroot/runtime/__init__.py:202` |

## Three access states

| State | Trigger | Error type |
|---|---|---|
| No cloud installed | `pip install scroot` only | `EnterpriseFeatureError` |
| Wrong plan | scroot-cloud installed, feature not in license.features | `PlanScopeError` |
| Expired / invalid license | token expired or forged | degrades to `EnterpriseFeatureError` |

## Invariants (enforced by CI)

- No `nacl`, `verify_license`, or `scroot_cloud` imports exist in `src/scroot/`.
- This file is regenerated in CI; a stale `SEAMS.md` fails the build.
