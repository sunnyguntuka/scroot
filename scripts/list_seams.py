#!/usr/bin/env python
"""Generate SEAMS.md: a human-readable audit of all gated seam surfaces.

Run from the repo root:
    python scripts/list_seams.py            # print to stdout
    python scripts/list_seams.py --write    # write SEAMS.md

CI freshness check:
    python scripts/list_seams.py --check    # exits 1 if SEAMS.md is stale
"""
from __future__ import annotations

import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src" / "scroot"
SEAMS_PATH = REPO_ROOT / "SEAMS.md"


def find_seam_call_sites() -> list[dict]:
    """Walk src/scroot/ AST and find all get_enterprise() call sites."""
    sites: list[dict] = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name != "get_enterprise":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            sites.append({
                "seam": first.value,
                "file": str(rel).replace("\\", "/"),
                "line": node.lineno,
            })
    return sites


def generate_seams_md() -> str:
    from scroot._messages import SEAM_LABELS, SEAM_OSS_COUNTERPART, DOCS_URL

    sites = find_seam_call_sites()
    by_seam: dict[str, list[str]] = {}
    for s in sites:
        by_seam.setdefault(s["seam"], []).append(f"`{s['file']}:{s['line']}`")

    lines = [
        "# SEAMS.md — scroot gated-surface audit",
        "",
        "> **Auto-generated** by `scripts/list_seams.py`. Do not edit by hand.",
        "> Re-generate: `python scripts/list_seams.py --write`",
        "",
        "Every row below is a place where scroot delegates to scroot-cloud.",
        "The OSS counterpart column shows what is available under Apache-2.0",
        "with no scroot-cloud installed. The open algorithm, calibration engine,",
        "local review UI, and air-gapped runtime are OSS. Only the operated,",
        "governed, audited lifecycle is gated.",
        "",
        f"Learn more: {DOCS_URL}",
        "",
        "| Seam key | Label | OSS counterpart | Call site(s) |",
        "|---|---|---|---|",
    ]

    for seam_key in sorted(SEAM_LABELS):
        label = SEAM_LABELS[seam_key]
        oss = SEAM_OSS_COUNTERPART.get(seam_key, "-")
        call_sites = ", ".join(by_seam.get(seam_key, ["-"]))
        lines.append(f"| `{seam_key}` | {label} | {oss} | {call_sites} |")

    lines += [
        "",
        "## Three access states",
        "",
        "| State | Trigger | Error type |",
        "|---|---|---|",
        "| No cloud installed | `pip install scroot` only | `EnterpriseFeatureError` |",
        "| Wrong plan | scroot-cloud installed, feature not in license.features | `PlanScopeError` |",
        "| Expired / invalid license | token expired or forged | degrades to `EnterpriseFeatureError` |",
        "",
        "## Invariants (enforced by CI)",
        "",
        "- No `nacl`, `verify_license`, or `scroot_cloud` imports exist in `src/scroot/`.",
        "- This file is regenerated in CI; a stale `SEAMS.md` fails the build.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--print"

    # Ensure src/scroot is importable
    sys.path.insert(0, str(REPO_ROOT / "src"))

    content = generate_seams_md()

    if mode == "--write":
        SEAMS_PATH.write_text(content, encoding="utf-8")
        print(f"Wrote {SEAMS_PATH}")
    elif mode == "--check":
        if not SEAMS_PATH.exists():
            print("ERROR: SEAMS.md does not exist. Run: python scripts/list_seams.py --write")
            sys.exit(1)
        committed = SEAMS_PATH.read_text(encoding="utf-8")
        if committed != content:
            print("ERROR: SEAMS.md is stale. Run: python scripts/list_seams.py --write")
            sys.exit(1)
        print("OK: SEAMS.md is up to date.")
    else:
        print(content)


if __name__ == "__main__":
    main()
