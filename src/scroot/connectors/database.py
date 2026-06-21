"""Database connector for scoring stored LLM responses.

Uses SQLAlchemy for database abstraction. Supports PostgreSQL, MySQL,
SQLite, BigQuery, Snowflake, and any SQLAlchemy-compatible backend.

Reads responses from a source table, scores them via Auditor, and writes
results to a result table. The result table is auto-created if absent.

Table and column names are validated against an allowlist pattern
(``^[A-Za-z_][A-Za-z0-9_]*$``). Identifiers that fail this pattern raise
``ValueError`` (not a warning). WHERE clauses and cursor columns are still
caller-controlled; only pass values you control.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger("scroot.connectors")


class SecurityWarning(Warning):
    """Warns about a potential security risk in connector configuration."""


_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _validate_identifier(name: str, label: str = "identifier") -> str:
    """Raise ValueError if ``name`` does not match the SQL identifier allowlist.

    Accepted pattern: ``^[A-Za-z_][A-Za-z0-9_]*$``.
    Rejects anything with spaces, quotes, semicolons, or special characters
    that could be used for SQL injection via identifier injection.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid SQL {label} {name!r}: only letters, digits, and "
            "underscores are allowed, and the name must start with a letter "
            "or underscore. Never pass user-supplied input as a table or "
            "column name."
        )
    return name


class DatabaseConnector:
    """Connector for scoring LLM responses stored in a SQL database.

    Args:
        connection_string: SQLAlchemy connection string.
            Examples:
                ``"postgresql://user:pass@host:5432/db"``
                ``"mysql+pymysql://user:pass@host/db"``
                ``"sqlite:///local.db"``
                ``"bigquery://project/dataset"``
        source_table: Name of the table containing LLM responses.
            Must match ``^[A-Za-z_][A-Za-z0-9_]*$``; raises ``ValueError``
            otherwise.
        column_map: Dict mapping entail field names to your column names.
            Required keys: ``"query"``, ``"response"``.
            Optional: ``"context"`` (JSON array or NULL), ``"id"`` (row identifier).
            All column name values must match the identifier allowlist.
        result_table: Table to write scores to. Auto-created if absent.
            Must match the identifier allowlist.
        batch_size: Rows fetched and scored per batch. Default 100.
        dry_run: If ``True``, ``fetch()`` and ``write_result()`` return the
            generated SQL strings instead of executing them. No I/O is
            performed. Useful for auditing the generated queries before
            running against a live database.
    """

    def __init__(
        self,
        connection_string: str,
        source_table: str,
        column_map: dict,
        result_table: str = "scroot_scores",
        batch_size: int = 100,
        dry_run: bool = False,
    ):
        try:
            import sqlalchemy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SQLAlchemy is required for database connectors: "
                "pip install 'scroot[database]'"
            ) from exc

        if "query" not in column_map or "response" not in column_map:
            raise ValueError("column_map must include 'query' and 'response' keys")

        # Validate all identifiers upfront: hard error, not a warning.
        _validate_identifier(source_table, "source_table")
        _validate_identifier(result_table, "result_table")
        for field_name, col_name in column_map.items():
            _validate_identifier(col_name, f"column_map[{field_name!r}]")

        self.connection_string = connection_string
        self.source_table = source_table
        self.column_map = column_map
        self.result_table = result_table
        self.batch_size = batch_size
        self.dry_run = dry_run

        if not dry_run:
            import sqlalchemy as sa
            self._engine = sa.create_engine(connection_string)
            self._metadata = sa.MetaData()
            self._ensure_result_table()
        else:
            self._engine = None
            self._metadata = None

    def _ensure_result_table(self) -> None:
        """Create the result table if it does not exist."""
        import sqlalchemy as sa

        if not sa.inspect(self._engine).has_table(self.result_table):
            table = sa.Table(
                self.result_table,
                self._metadata,
                sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                sa.Column("source_row_id", sa.String(255), index=True),
                sa.Column("scored_at", sa.DateTime),
                sa.Column("iqs", sa.Float),
                sa.Column("groundedness", sa.Float, nullable=True),
                sa.Column("completeness", sa.Float),
                sa.Column("relevance", sa.Float),
                sa.Column("consistency", sa.Float),
                sa.Column("confidence", sa.Float),
                sa.Column("flags", sa.Text),
                sa.Column("details", sa.Text),
                sa.Column("strategy", sa.String(50), nullable=True),
                sa.Column("sample_seed", sa.Integer, nullable=True),
            )
            table.create(self._engine)
            logger.info("Created result table: %s", self.result_table)

    def _validate_write_schema(self) -> None:
        """Verify the result table has the expected columns before writing."""
        import sqlalchemy as sa
        inspector = sa.inspect(self._engine)
        if not inspector.has_table(self.result_table):
            return  # _ensure_result_table will create it
        existing = {col["name"] for col in inspector.get_columns(self.result_table)}
        required = {"source_row_id", "scored_at", "iqs", "completeness",
                    "relevance", "consistency", "confidence", "flags", "details"}
        missing = required - existing
        if missing:
            raise ValueError(
                f"Result table {self.result_table!r} is missing required columns: "
                f"{sorted(missing)}. Run the schema migration or drop and re-create "
                "the table to apply the current schema."
            )

    def _parse_context(self, raw_value) -> list[str] | None:
        """Parse a context column value into a list of strings."""
        if raw_value is None:
            return None
        if isinstance(raw_value, list):
            return [str(c) for c in raw_value]
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
                if isinstance(parsed, list):
                    return [str(c) for c in parsed]
            except json.JSONDecodeError:
                return [raw_value]
        return None

    def fetch(
        self,
        limit: int | None = None,
        where: str | None = None,
        offset: int = 0,
        stream: bool = False,
    ) -> "list[dict] | str":
        """Fetch rows from the source table.

        Args:
            limit: Max rows to fetch. ``None`` fetches all rows.
            where: Optional SQL WHERE clause (without the ``WHERE`` keyword).
                This string is injected verbatim; only pass values you
                control, never user input.
            offset: Row offset for pagination.
            stream: If ``True``, use a server-side streaming cursor
                (``stream_results=True``) to avoid loading all rows into
                memory. Rows are still returned as a list; the streaming
                happens internally. Recommended for large tables.

        Returns:
            List of dicts with entail field keys plus ``"_row_id"`` and
            ``"_raw"``. Returns the SQL string instead when ``dry_run=True``.
        """
        parts = [f"SELECT * FROM {self.source_table}"]
        if where:
            parts.append(f"WHERE {where}")
        if limit is not None:
            parts.append(f"LIMIT {limit}")
        if offset:
            parts.append(f"OFFSET {offset}")
        sql = " ".join(parts)

        if self.dry_run:
            return sql

        import sqlalchemy as sa
        connect_kwargs: dict = {}
        if stream:
            connect_kwargs["execution_options"] = {"stream_results": True}

        with self._engine.connect() as conn:
            if stream:
                conn = conn.execution_options(stream_results=True)
            cursor = conn.execute(sa.text(sql))
            col_names = list(cursor.keys())
            rows = []
            id_col = self.column_map.get("id", "id")
            for row in cursor:
                raw = dict(zip(col_names, row))
                mapped: dict = {}
                for entail_field, db_col in self.column_map.items():
                    mapped[entail_field] = raw.get(db_col)
                mapped["_row_id"] = raw.get(id_col, raw.get("id"))
                mapped["_raw"] = raw
                rows.append(mapped)
            return rows

    def write_result(
        self,
        row_id,
        result,
        strategy: str | None = None,
        seed: int | None = None,
    ) -> "None | str":
        """Write a single EntailmentResult to the result table.

        Args:
            row_id: Source row identifier.
            result: EntailmentResult from Auditor.score().
            strategy: Optional sampling strategy label.
            seed: Optional sampling seed.

        Returns:
            ``None`` normally; the SQL INSERT string when ``dry_run=True``.
        """
        insert_sql = (
            f"INSERT INTO {self.result_table} "
            "(source_row_id, scored_at, iqs, groundedness, completeness, "
            "relevance, consistency, confidence, flags, details, strategy, sample_seed) "
            "VALUES "
            "(:source_row_id, :scored_at, :iqs, :groundedness, :completeness, "
            ":relevance, :consistency, :confidence, :flags, :details, :strategy, :seed)"
        )

        if self.dry_run:
            return insert_sql

        self._validate_write_schema()

        import sqlalchemy as sa
        with self._engine.connect() as conn:
            conn.execute(sa.text(insert_sql), {
                "source_row_id": str(row_id),
                "scored_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "iqs": result.iqs,
                "groundedness": result.groundedness,
                "completeness": result.completeness,
                "relevance": result.relevance,
                "consistency": result.consistency,
                "confidence": result.confidence,
                "flags": json.dumps(result.flags),
                "details": json.dumps(result.details, default=str),
                "strategy": strategy,
                "seed": seed,
            })
            conn.commit()
        return None

    def score_all(self, auditor, where: str | None = None) -> dict:
        """Score all responses in the source table.

        Args:
            auditor: Auditor instance.
            where: Optional SQL WHERE clause to filter rows.

        Returns:
            Dict with total_scored, mean_iqs, flag_counts.
        """
        offset = 0
        total_scored = 0
        all_iqs: list[float] = []
        all_flags: list[str] = []

        while True:
            rows = self.fetch(limit=self.batch_size, where=where, offset=offset)
            if not rows:
                break
            for row in rows:
                context = self._parse_context(row.get("context"))
                result = auditor.score(
                    query=str(row["query"]),
                    response=str(row["response"]),
                    context=context,
                )
                self.write_result(row["_row_id"], result)
                all_iqs.append(result.iqs)
                all_flags.extend(result.flags)
                total_scored += 1
            offset += self.batch_size
            logger.info("Scored %d responses so far...", total_scored)

        import numpy as np
        flag_counts: dict[str, int] = {}
        for f in all_flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1

        return {
            "total_scored": total_scored,
            "mean_iqs": float(np.mean(all_iqs)) if all_iqs else 0.0,
            "flag_counts": flag_counts,
        }

    def score_where(self, auditor, where: str) -> dict:
        """Score responses matching a SQL WHERE clause.

        Args:
            auditor: Auditor instance.
            where: SQL WHERE clause (without the WHERE keyword).

        Returns:
            Dict with total_scored, mean_iqs, flag_counts.
        """
        return self.score_all(auditor, where=where)

    def score_sampled(
        self,
        auditor,
        strategy: str = "random",
        sample_size: int | None = None,
        sample_pct: float | None = None,
        seed: int | None = 42,
        where: str | None = None,
    ):
        """Score a sampled subset of responses from the database.

        Fetches all matching rows, applies sampling, scores the sample,
        and writes results back with the sampling strategy label.

        Args:
            auditor: Auditor instance.
            strategy: Sampling strategy ("random", "percentage", "confidence").
            sample_size: For "random" strategy.
            sample_pct: For "percentage" strategy.
            seed: Random seed.
            where: Optional SQL WHERE filter.

        Returns:
            SamplingResult.
        """
        from ..sampling import sample_and_score

        rows = self.fetch(where=where)
        items = []
        for row in rows:
            context = self._parse_context(row.get("context"))
            items.append({
                "query": str(row["query"]),
                "response": str(row["response"]),
                "context": context,
                "_row_id": row["_row_id"],
            })

        result = sample_and_score(
            auditor=auditor,
            items=items,
            strategy=strategy,
            sample_size=sample_size,
            sample_pct=sample_pct,
            seed=seed,
        )

        for si in result.scored_items:
            row_id = si["item"].get("_row_id")
            if row_id is not None:
                self.write_result(row_id, si["result"], strategy=strategy, seed=seed)

        return result

    def score_incremental(
        self, auditor, cursor_column: str = "created_at"
    ) -> dict:
        """Score only new responses since the last scored row.

        Finds the maximum cursor_column value among source rows that have
        already been scored, then scores all source rows where
        cursor_column > that watermark.

        Args:
            auditor: Auditor instance.
            cursor_column: Column in source_table to use as the cursor.
                Must match the identifier allowlist.

        Returns:
            Dict with total_scored, mean_iqs, flag_counts.
        """
        _validate_identifier(cursor_column, "cursor_column")

        import sqlalchemy as sa

        id_col = self.column_map.get("id", "id")
        with self._engine.connect() as conn:
            max_cursor = conn.execute(sa.text(
                f"SELECT MAX(src.{cursor_column}) FROM {self.source_table} src "
                f"WHERE CAST(src.{id_col} AS TEXT) IN "
                f"(SELECT source_row_id FROM {self.result_table})"
            )).scalar()

        where = f"{cursor_column} > '{max_cursor}'" if max_cursor else None
        return self.score_all(auditor, where=where)
