# scroot versioning policy

## SemVer contract

scroot follows [Semantic Versioning 2.0](https://semver.org/). The meaningful
boundaries for scroot-cloud compatibility are:

| Change type | Version bump | scroot-cloud pin required? |
|---|---|---|
| Add a new seam key | **minor** | Yes — new seam is only available on scroot ≥ X.Y |
| Remove or rename a seam key | **major** | Yes — cloud breaks on old scroot |
| Add / change public Python API (non-seam) | minor | No |
| Bug fix, doc change, internal refactor | patch | No |
| Breaking change to public API | major | Yes |

## The seam API

A **seam key** (e.g. `"audit.export"`) is the unit of compatibility between
`scroot` and `scroot-cloud`. Keys are declared in `src/scroot/_messages.py`
(`SEAM_LABELS`) and scanned into `SEAMS.md` by `scripts/list_seams.py`.

Rules:
1. **Never rename a seam key without a major bump.** `scroot-cloud` pins seam
   names in its `register()` function. A renamed key breaks the plugin at
   runtime with `EnterpriseFeatureError` rather than an import error, which is
   surprising.
2. **Adding a seam is always minor.** The new seam raises `EnterpriseFeatureError`
   on existing cloud installs that don't supply it yet (gracefully degraded,
   not broken).
3. **Removing a seam is always major.** The cloud plugin would try to register a
   key that no longer exists.

## How scroot-cloud pins scroot

`scroot-cloud/pyproject.toml` pins:
```
scroot>=0.3,<1.0
```

For each minor release of `scroot` that adds new seams, `scroot-cloud` ships a
companion release (same minor, patch 0) that registers the new seams. The CI
contract (see below) catches drift.

## CI contract

### scroot CI (`.github/workflows/tests.yml`)

```yaml
- name: Assert every seam raises EnterpriseFeatureError without cloud
  run: python -c "..."
```

Asserts: for every key in `SEAM_LABELS`, calling `get_enterprise(key)` on a
fresh registry (no cloud plugin loaded) raises `EnterpriseFeatureError`. This
proves the OSS upsell path is wired for every seam.

### scroot-cloud CI (`.github/workflows/tests.yml`)

```yaml
- name: Assert register() covers all scroot seams
  run: python -c "..."
```

Asserts: the set of seam keys registered by `scroot_cloud.register()` equals
`scroot._messages.SEAM_LABELS.keys()`. Any mismatch (seam added to scroot but
not cloud, or cloud registering a phantom key) fails CI.

### Scheduled compatibility job (`scroot-cloud/.github/workflows/compat.yml`)

Runs daily against the latest published `scroot` to catch incompatibilities
introduced by patch releases. Triggers an issue if it fails.

## Upgrade path for breaking changes

When a major version bump is required:
1. Tag the current release `vX.Y.Z` and push.
2. Open a migration issue listing every renamed/removed seam.
3. Coordinate a matching major bump in `scroot-cloud` on the same day.
4. Update `scroot-cloud/pyproject.toml` to `scroot>=X+1,<X+2`.
5. Update the compatibility CI job to test the new major.
