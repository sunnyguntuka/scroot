"""Tests for entail.connectors.DatabaseConnector using in-memory SQLite."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")

from scroot.connectors import DatabaseConnector  # noqa: E402
from scroot.result import EntailmentResult  # noqa: E402


def _fake_result(iqs: float = 0.85, flags: list[str] | None = None) -> EntailmentResult:
    return EntailmentResult(
        groundedness=0.85,
        completeness=0.85,
        relevance=0.85,
        consistency=0.85,
        confidence=0.85,
        iqs=iqs,
        flags=flags or [],
        details={"note": "mock"},
    )


def _mock_auditor(iqs: float = 0.85):
    auditor = MagicMock()
    auditor.score.return_value = _fake_result(iqs=iqs)
    return auditor


def _sqlite_connector(db_url: str, source_table: str = "responses", column_map: dict | None = None):
    return DatabaseConnector(
        connection_string=db_url,
        source_table=source_table,
        column_map=column_map or {"query": "query", "response": "response"},
    )


def _populate_table(engine, table: str, rows: list[dict]) -> None:
    import sqlalchemy as sa
    with engine.connect() as conn:
        for row in rows:
            cols = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            conn.execute(sa.text(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"), row)
        conn.commit()


def _create_source_table(engine, table: str, extra_cols: list[str] | None = None) -> None:
    import sqlalchemy as sa
    col_defs = "id INTEGER PRIMARY KEY, query TEXT, response TEXT"
    if extra_cols:
        col_defs += ", " + ", ".join(extra_cols)
    with engine.connect() as conn:
        conn.execute(sa.text(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs})"))
        conn.commit()


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path}/test.db"


@pytest.fixture
def engine(db_url):
    import sqlalchemy as sa
    return sa.create_engine(db_url)


# --- result table auto-creation ---

def test_result_table_auto_created(db_url, engine):
    import sqlalchemy as sa
    _create_source_table(engine, "responses")
    _sqlite_connector(db_url)
    assert sa.inspect(engine).has_table("scroot_scores")


# --- validation ---

def test_missing_query_column_raises(db_url, engine):
    _create_source_table(engine, "responses")
    with pytest.raises(ValueError, match="column_map must include"):
        DatabaseConnector(
            connection_string=db_url,
            source_table="responses",
            column_map={"response": "response"},
        )


def test_missing_response_column_raises(db_url, engine):
    _create_source_table(engine, "responses")
    with pytest.raises(ValueError, match="column_map must include"):
        DatabaseConnector(
            connection_string=db_url,
            source_table="responses",
            column_map={"query": "query"},
        )


# --- fetch ---

def test_fetch_returns_rows(db_url, engine):
    _create_source_table(engine, "responses")
    _populate_table(engine, "responses", [
        {"id": 1, "query": "q1", "response": "r1"},
        {"id": 2, "query": "q2", "response": "r2"},
    ])
    connector = _sqlite_connector(db_url)
    rows = connector.fetch()
    assert len(rows) == 2
    assert rows[0]["query"] == "q1"


def test_fetch_with_limit(db_url, engine):
    _create_source_table(engine, "responses")
    _populate_table(engine, "responses", [
        {"id": i, "query": f"q{i}", "response": f"r{i}"} for i in range(10)
    ])
    connector = _sqlite_connector(db_url)
    rows = connector.fetch(limit=3)
    assert len(rows) == 3


def test_fetch_with_offset(db_url, engine):
    _create_source_table(engine, "responses")
    _populate_table(engine, "responses", [
        {"id": i, "query": f"q{i}", "response": f"r{i}"} for i in range(10)
    ])
    connector = _sqlite_connector(db_url)
    rows = connector.fetch(limit=5, offset=5)
    assert len(rows) == 5


def test_fetch_with_where(db_url, engine):
    _create_source_table(engine, "responses", extra_cols=["agent TEXT"])
    _populate_table(engine, "responses", [
        {"id": 1, "query": "q1", "response": "r1", "agent": "bot_a"},
        {"id": 2, "query": "q2", "response": "r2", "agent": "bot_b"},
        {"id": 3, "query": "q3", "response": "r3", "agent": "bot_a"},
    ])
    connector = _sqlite_connector(db_url)
    rows = connector.fetch(where="agent = 'bot_a'")
    assert len(rows) == 2


# --- column map remapping ---

def test_column_map_remapping(db_url, engine):
    import sqlalchemy as sa
    with engine.connect() as conn:
        conn.execute(sa.text(
            "CREATE TABLE custom (id INTEGER PRIMARY KEY, user_q TEXT, agent_r TEXT)"
        ))
        conn.execute(sa.text("INSERT INTO custom VALUES (1, 'hello', 'world')"))
        conn.commit()
    connector = DatabaseConnector(
        connection_string=db_url,
        source_table="custom",
        column_map={"query": "user_q", "response": "agent_r"},
    )
    rows = connector.fetch()
    assert rows[0]["query"] == "hello"
    assert rows[0]["response"] == "world"


# --- context parsing ---

def test_context_json_parsing(db_url, engine):
    _create_source_table(engine, "responses", extra_cols=["context TEXT"])
    _populate_table(engine, "responses", [
        {"id": 1, "query": "q", "response": "r", "context": '["chunk1", "chunk2"]'},
    ])
    connector = DatabaseConnector(
        connection_string=db_url,
        source_table="responses",
        column_map={"query": "query", "response": "response", "context": "context"},
    )
    rows = connector.fetch()
    parsed = connector._parse_context(rows[0]["context"])
    assert parsed == ["chunk1", "chunk2"]


def test_context_null_handling(db_url):
    connector = _sqlite_connector(db_url)
    # engine/table creation happens in __init__ already
    assert connector._parse_context(None) is None


def test_context_plain_string_wrapped(db_url):
    connector = _sqlite_connector(db_url)
    result = connector._parse_context("not json")
    assert result == ["not json"]


# --- score_all ---

def test_sqlite_score_all(db_url, engine):
    import sqlalchemy as sa
    _create_source_table(engine, "responses")
    _populate_table(engine, "responses", [
        {"id": i, "query": f"q{i}", "response": f"r{i}"} for i in range(10)
    ])
    auditor = _mock_auditor()
    connector = _sqlite_connector(db_url)
    stats = connector.score_all(auditor)
    assert stats["total_scored"] == 10
    assert abs(stats["mean_iqs"] - 0.85) < 1e-6
    # Verify result table was written
    with engine.connect() as conn:
        count = conn.execute(sa.text("SELECT COUNT(*) FROM scroot_scores")).scalar()
    assert count == 10


def test_score_all_with_where(db_url, engine):
    _create_source_table(engine, "responses", extra_cols=["tag TEXT"])
    _populate_table(engine, "responses", [
        {"id": 1, "query": "q1", "response": "r1", "tag": "keep"},
        {"id": 2, "query": "q2", "response": "r2", "tag": "skip"},
        {"id": 3, "query": "q3", "response": "r3", "tag": "keep"},
    ])
    auditor = _mock_auditor()
    connector = _sqlite_connector(db_url)
    stats = connector.score_all(auditor, where="tag = 'keep'")
    assert stats["total_scored"] == 2


# --- score_sampled ---

def test_sqlite_score_sampled(db_url, engine):
    import sqlalchemy as sa
    _create_source_table(engine, "responses")
    _populate_table(engine, "responses", [
        {"id": i, "query": f"q{i}", "response": f"r{i}"} for i in range(100)
    ])
    auditor = _mock_auditor()
    connector = _sqlite_connector(db_url)
    result = connector.score_sampled(auditor, strategy="random", sample_size=10, seed=42)
    assert result.sample_size == 10
    with engine.connect() as conn:
        count = conn.execute(sa.text("SELECT COUNT(*) FROM scroot_scores")).scalar()
    assert count == 10


# --- write_result / roundtrip ---

def test_write_result_roundtrip(db_url, engine):
    import sqlalchemy as sa
    _create_source_table(engine, "responses")
    _populate_table(engine, "responses", [{"id": 1, "query": "q", "response": "r"}])
    connector = _sqlite_connector(db_url)
    fake = _fake_result(iqs=0.77)
    connector.write_result(row_id=1, result=fake, strategy="random", seed=42)
    with engine.connect() as conn:
        row = conn.execute(sa.text("SELECT * FROM scroot_scores WHERE source_row_id='1'")).fetchone()
    assert row is not None
    cols = ["id", "source_row_id", "scored_at", "iqs", "groundedness",
            "completeness", "relevance", "consistency", "confidence",
            "flags", "details", "strategy", "sample_seed"]
    row_dict = dict(zip(cols, row))
    assert abs(row_dict["iqs"] - 0.77) < 1e-6
    assert row_dict["strategy"] == "random"
    assert row_dict["sample_seed"] == 42
    assert json.loads(row_dict["flags"]) == []


# --- incremental scoring ---

def test_incremental_scoring(db_url, engine):
    import sqlalchemy as sa
    _create_source_table(engine, "responses", extra_cols=["created_at TEXT"])
    _populate_table(engine, "responses", [
        {"id": i, "query": f"q{i}", "response": f"r{i}", "created_at": f"2026-05-{i+1:02d}"}
        for i in range(8)
    ])
    auditor = _mock_auditor()
    connector = _sqlite_connector(db_url)
    # First run: score rows 0-4
    connector.score_all(auditor, where="created_at <= '2026-05-05'")
    with engine.connect() as conn:
        first_count = conn.execute(sa.text("SELECT COUNT(*) FROM scroot_scores")).scalar()
    assert first_count == 5
    # Incremental: should pick up rows 5-7 (created_at > max scored_at)
    stats = connector.score_incremental(auditor, cursor_column="created_at")
    assert stats["total_scored"] == 3
    with engine.connect() as conn:
        total = conn.execute(sa.text("SELECT COUNT(*) FROM scroot_scores")).scalar()
    assert total == 8
