"""Security tests for FeedbackStore: path traversal, encryption,
field masking, delete/purge, max_records, ttl_days, integrity (C-3/C-4/H-1/H-7)."""

import pytest
import warnings
from datetime import datetime, timezone, timedelta

from scroot.feedback.store import FeedbackStore, CorrectionRecord, _compute_record_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(id="r1", query="What is the policy?", timestamp=None):
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    return CorrectionRecord(
        id=id,
        timestamp=ts,
        query=query,
        response="Wrong answer",
        scores={"iqs": 0.3},
        flags=["hallucination_risk"],
        correction="Correct answer",
        reason="Made up facts",
        context_used=["Real context"],
        corrected_by="human",
    )


# ---------------------------------------------------------------------------
# H-1: Path traversal
# ---------------------------------------------------------------------------

def test_path_traversal_relative_dots_rejected():
    with pytest.raises(ValueError, match="traversal"):
        FeedbackStore("../../etc/cron.d/backdoor.jsonl")


def test_path_must_end_with_jsonl():
    with pytest.raises(ValueError, match=r"\.jsonl"):
        FeedbackStore("/tmp/mystore.txt")


def test_valid_absolute_path_accepted(tmp_path):
    store = FeedbackStore(str(tmp_path / "ok.jsonl"))
    assert store.path.endswith(".jsonl")


def test_relative_path_without_traversal_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = FeedbackStore("mystore.jsonl")
    assert store.path.endswith(".jsonl")


# ---------------------------------------------------------------------------
# C-3: Unencrypted warning
# ---------------------------------------------------------------------------

def test_unencrypted_warning_on_first_add(tmp_path):
    store = FeedbackStore(str(tmp_path / "w.jsonl"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.add(make_record())
    messages = [str(w.message) for w in caught]
    assert any("unencrypted" in m.lower() for m in messages)


def test_unencrypted_warning_fires_only_once(tmp_path):
    store = FeedbackStore(str(tmp_path / "w2.jsonl"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.add(make_record("r1"))
        store.add(make_record("r2"))
    unencrypted_warns = [w for w in caught if "unencrypted" in str(w.message).lower()]
    assert len(unencrypted_warns) == 1


# ---------------------------------------------------------------------------
# C-3: Encryption
# ---------------------------------------------------------------------------

def test_encryption_roundtrip(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    store = FeedbackStore(str(tmp_path / "enc.jsonl"), encryption_key=key)
    record = make_record(query="SSN: 078-05-1120")
    store.add(record)

    # Raw file must not contain the plaintext query
    raw = (tmp_path / "enc.jsonl").read_text(encoding="utf-8")
    assert "SSN" not in raw
    assert "078-05-1120" not in raw

    # But decrypted read returns the original
    records = store.get_all()
    assert len(records) == 1
    assert records[0].query == "SSN: 078-05-1120"


def test_encrypted_store_no_unencrypted_warning(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    store = FeedbackStore(str(tmp_path / "enc2.jsonl"), encryption_key=key)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.add(make_record())
    unencrypted_warns = [w for w in caught if "unencrypted" in str(w.message).lower()]
    assert len(unencrypted_warns) == 0


def test_str_encryption_key_accepted(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet
    key = Fernet.generate_key().decode()
    store = FeedbackStore(str(tmp_path / "enc3.jsonl"), encryption_key=key)
    store.add(make_record())
    assert store.count() == 1


# ---------------------------------------------------------------------------
# C-3: Field masking
# ---------------------------------------------------------------------------

def test_field_mask_redacts_on_disk(tmp_path):
    store = FeedbackStore(
        str(tmp_path / "masked.jsonl"),
        field_mask=["query", "context_used"],
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record(query="My SSN is 078-05-1120"))
    raw = (tmp_path / "masked.jsonl").read_text(encoding="utf-8")
    assert "078-05-1120" not in raw
    assert "[REDACTED]" in raw


def test_field_mask_non_masked_fields_intact(tmp_path):
    store = FeedbackStore(
        str(tmp_path / "masked2.jsonl"),
        field_mask=["query"],
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record(query="sensitive", id="keep-me"))
    records = store.get_all()
    assert records[0].id == "keep-me"
    assert records[0].query == "[REDACTED]"


# ---------------------------------------------------------------------------
# C-4: delete()
# ---------------------------------------------------------------------------

def test_delete_removes_record(tmp_path):
    store = FeedbackStore(str(tmp_path / "del.jsonl"))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record("r1"))
        store.add(make_record("r2"))
        store.add(make_record("r3"))
    deleted = store.delete("r2")
    assert deleted == 1
    ids = [r.id for r in store.get_all()]
    assert "r2" not in ids
    assert "r1" in ids
    assert "r3" in ids


def test_delete_nonexistent_returns_zero(tmp_path):
    store = FeedbackStore(str(tmp_path / "del2.jsonl"))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record("r1"))
    assert store.delete("does-not-exist") == 0


# ---------------------------------------------------------------------------
# C-4: purge()
# ---------------------------------------------------------------------------

def test_purge_removes_old_records(tmp_path):
    store = FeedbackStore(str(tmp_path / "purge.jsonl"))
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record("old", timestamp=old_ts))
        store.add(make_record("new", timestamp=new_ts))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    removed = store.purge(before_date=cutoff)
    assert removed == 1
    ids = [r.id for r in store.get_all()]
    assert "old" not in ids
    assert "new" in ids


# ---------------------------------------------------------------------------
# C-4: max_records
# ---------------------------------------------------------------------------

def test_max_records_prunes_oldest(tmp_path):
    store = FeedbackStore(str(tmp_path / "maxr.jsonl"), max_records=3)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        for i in range(5):
            store.add(make_record(f"r{i}"))
    records = store.get_all()
    assert len(records) == 3
    ids = [r.id for r in records]
    assert "r4" in ids  # most recent kept
    assert "r0" not in ids  # oldest pruned


# ---------------------------------------------------------------------------
# C-4: ttl_days
# ---------------------------------------------------------------------------

def test_ttl_days_filters_expired(tmp_path):
    store = FeedbackStore(str(tmp_path / "ttl.jsonl"), ttl_days=7)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record("old", timestamp=old_ts))
        store.add(make_record("new", timestamp=new_ts))
    records = store.get_all()
    ids = [r.id for r in records]
    assert "old" not in ids
    assert "new" in ids


# ---------------------------------------------------------------------------
# H-7: Record hash + integrity
# ---------------------------------------------------------------------------

def test_record_hash_is_set_on_add(tmp_path):
    store = FeedbackStore(str(tmp_path / "hash.jsonl"))
    record = make_record()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(record)
    loaded = store.get_all()[0]
    assert loaded.record_hash is not None
    assert len(loaded.record_hash) == 16


def test_record_number_is_sequential(tmp_path):
    store = FeedbackStore(str(tmp_path / "seq.jsonl"))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record("r1"))
        store.add(make_record("r2"))
        store.add(make_record("r3"))
    records = store.get_all()
    numbers = [r.record_number for r in records]
    assert numbers == [1, 2, 3]


def test_validate_integrity_clean_store(tmp_path):
    store = FeedbackStore(str(tmp_path / "int.jsonl"))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record("r1"))
        store.add(make_record("r2"))
    issues = store.validate_integrity()
    assert issues == []


def test_validate_integrity_detects_hash_tamper(tmp_path):
    path = tmp_path / "tamper.jsonl"
    store = FeedbackStore(str(path))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        store.add(make_record("r1"))

    # Manually corrupt the stored hash
    import json
    lines = path.read_text(encoding="utf-8").splitlines()
    data = json.loads(lines[0])
    data["record_hash"] = "deadbeef0000"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    issues = store.validate_integrity()
    assert any("mismatch" in issue.lower() for issue in issues)


# ---------------------------------------------------------------------------
# H-7: compute_record_hash determinism
# ---------------------------------------------------------------------------

def test_compute_record_hash_deterministic():
    r = make_record("x", query="test query")
    h1 = _compute_record_hash(r)
    h2 = _compute_record_hash(r)
    assert h1 == h2


def test_compute_record_hash_changes_with_content():
    r1 = make_record("x", query="query A")
    r2 = make_record("x", query="query B")
    assert _compute_record_hash(r1) != _compute_record_hash(r2)
