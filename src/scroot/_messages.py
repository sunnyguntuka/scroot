# Apache-2.0. Central catalog for gated-surface labels and docs URLs.
# EnterpriseFeatureError and _entitlements.get_enterprise use these so that
# all upsell copy lives in one place and is edited once.
from __future__ import annotations

DOCS_URL = "https://scroot.dev/cloud"

# Human-readable label for each gated seam key.
# Key: seam name used in get_enterprise() / register_enterprise().
# Value: label shown in EnterpriseFeatureError and PlanScopeError messages.
SEAM_LABELS: dict[str, str] = {
    "audit.export":         "Audit evidence export",
    "calibration.schedule": "Managed calibration lifecycle",
    "pii.policy":           "Regulatory PII policy management",
    "runtime.managed":      "Managed scoring runtime",
    "metrics.builder":      "No-code custom metric builder",
    "review.queue":         "Hosted review queue",
    "drift.continuous":     "Continuous drift monitoring (Ampulla)",
}

# OSS counterpart for each seam (what is available without scroot-cloud).
# Used by SEAMS.md generation and the public seam-audit doc.
SEAM_OSS_COUNTERPART: dict[str, str] = {
    "audit.export":         "result.evidence_map (scoring data, no signing/retention)",
    "calibration.schedule": "calibrate() fit + CalibrationResult",
    "pii.policy":           "scrub() local masking with allowlist + grounding mode",
    "runtime.managed":      "runtime.run() local air-gapped scoring + preflight()",
    "metrics.builder":      "register_metric() code-level custom metric",
    "review.queue":         "review.ui() local single-user viewer (scroot serve)",
    "drift.continuous":     "regression_check() point-in-time CI gate",
}
