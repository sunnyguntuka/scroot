# scroot pricing & plan tiers

> This page is derived from `scroot_cloud/plans.py`. The feature matrix in code
> and in this doc are the same source.

## The open-core promise

The scoring engine, calibration algorithm, local review UI, numeric grounding
verifier, and air-gapped runtime are **free and open** (Apache-2.0). You can
self-host, audit, and fork them. There is no crippled-OSS model: every OSS
surface is fully functional with no license and no scroot-cloud installed.

Enterprise is the **operated, governed, audited layer** around the open core:
signed evidence bundles for compliance, managed calibration lifecycle, hosted
multi-reviewer queues, SLA-backed runtimes, and continuous drift monitoring.

---

## Plans

### Open Source — free forever

| Feature | Available |
|---|---|
| NLI scoring engine (IQS, groundedness, completeness, relevance, consistency, confidence) | Yes |
| Evidence map (sentence-level grounding breakdown) | Yes |
| `calibrate()` — fit threshold + weights from labeled data | Yes |
| `regression_check()` — point-in-time CI quality gate | Yes |
| `register_metric()` — code-level custom metrics | Yes |
| `scrub()` — local PII masking with allowlist + grounding mode | Yes |
| `runtime.run()` — local air-gapped scoring runtime | Yes |
| `runtime.preflight()` — model cache status check | Yes |
| `review.ui()` — local single-user Review Console (`scroot serve`) | Yes |
| Numeric grounding verifier | Yes |
| Batch scoring, eval suites, sampling | Yes |

---

### Team

Includes everything in Open Source, plus:

| Feature | Seam key |
|---|---|
| Hosted multi-reviewer queue (assignment, claim/lock, sign-off) | `review.queue` |
| No-code visual metric builder | `metrics.builder` |

---

### Enterprise

Includes everything in Team, plus:

| Feature | Seam key |
|---|---|
| Signed, retained audit evidence bundles (SOC II / compliance) | `audit.export` |
| Regulatory PII policy management (GDPR/HIPAA presets, DLP hooks) | `pii.policy` |
| Managed scoring runtime (autoscaling, SLA, health checks) | `runtime.managed` |
| Managed calibration lifecycle (scheduled, versioned, audit-grade) | `calibration.schedule` |
| Continuous drift monitoring via Ampulla | `drift.continuous` |

---

### Enterprise (air-gapped)

Same feature set as Enterprise. The runtime is offline-licensed via an embedded
Ed25519 public key — no network calls during license verification. Designed for
regulated environments where all traffic must remain on-premises.

---

## Feature-to-seam mapping

The table below maps plan features to the seam key used in `scroot_cloud/plans.py`
and enforced by the scoping layer (`PlanScopeError`).

| Seam key | Team | Enterprise | Enterprise (air-gapped) |
|---|:---:|:---:|:---:|
| `review.queue` | Yes | Yes | Yes |
| `metrics.builder` | Yes | Yes | Yes |
| `audit.export` | - | Yes | Yes |
| `pii.policy` | - | Yes | Yes |
| `runtime.managed` | - | Yes | Yes |
| `calibration.schedule` | - | Yes | Yes |
| `drift.continuous` | - | Yes | Yes |

For current pricing, contact [scroot.dev/cloud](https://scroot.dev/cloud).
