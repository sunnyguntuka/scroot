"""Snowflake Cortex integration: score Cortex LLM responses with scroot.

Two pieces work together:

1. **Inside Snowflake** (reference only -- see `SNOWPARK_UDF_SQL` below):
   a Snowpark Python UDF wraps `auditor.score()` and is called as part of
   a SQL pipeline immediately after `SNOWFLAKE.CORTEX.COMPLETE(...)`,
   writing query/response/context/scores into a results table.

2. **From Python** (runnable below): scroot's `DatabaseConnector` connects
   to that results table and re-scores / audits it, the same way it would
   connect to Postgres, MySQL, or any other SQLAlchemy-supported database.
   Swap the SQLite connection string for a Snowflake one:

       "snowflake://<user>:<password>@<account>/<database>/<schema>"
       "?warehouse=<warehouse>&role=<role>"

   (requires `pip install snowflake-sqlalchemy`)

This example uses an in-memory SQLite database so it runs with no external
services, demonstrating the connector pattern end-to-end.
"""

from __future__ import annotations

import tempfile
import warnings

from scroot import Auditor
from scroot.connectors import DatabaseConnector, SecurityWarning

# --- Reference: Snowpark Python UDF that scores Cortex completions in-warehouse ---
#
# CREATE OR REPLACE FUNCTION scroot_score(query STRING, response STRING, context ARRAY)
# RETURNS VARIANT
# LANGUAGE PYTHON
# RUNTIME_VERSION = '3.11'
# PACKAGES = ('scroot')
# HANDLER = 'score_handler'
# AS
# $$
# from scroot import Auditor
# _auditor = Auditor()
#
# def score_handler(query, response, context):
#     result = _auditor.score(query=query, response=response, context=context)
#     return result.to_dict()
# $$;
#
# -- Use it after a Cortex completion:
# INSERT INTO scroot_scores (query, response, context, scores)
# SELECT
#     prompt,
#     SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', prompt) AS response,
#     context_chunks,
#     scroot_score(prompt, response, context_chunks)
# FROM cortex_inference_log;
SNOWPARK_UDF_SQL = __doc__  # documentation only, not executed

# --- Runnable: pull responses from a table and audit them with scroot ---

import sqlalchemy as sa  # noqa: E402

db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
db_url = f"sqlite:///{db_path}"

engine = sa.create_engine(db_url)
with engine.begin() as conn:
    conn.execute(sa.text(
        "CREATE TABLE cortex_responses ("
        "id INTEGER PRIMARY KEY, prompt TEXT, completion TEXT, context_json TEXT)"
    ))
    conn.execute(sa.text(
        "INSERT INTO cortex_responses (prompt, completion, context_json) VALUES "
        "(:prompt, :completion, :context_json)"
    ), [
        {
            "prompt": "What is our refund policy?",
            "completion": "We offer a 30-day full refund at no extra cost.",
            "context_json": (
                '["All customers are eligible for a 30-day full refund '
                'at no extra cost.", '
                '"Refund requests must be submitted via the support portal."]'
            ),
        },
        {
            "prompt": "What is our refund policy?",
            "completion": "We offer a 90-day money-back guarantee with free shipping.",
            "context_json": (
                '["All customers are eligible for a 30-day full refund '
                'at no extra cost."]'
            ),
        },
    ])

# DatabaseConnector warns that table/column names in `source_table` and
# `result_table` are not parameterised -- only pass trusted, code-controlled
# values (see docs/security.md).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SecurityWarning)
    connector = DatabaseConnector(
        connection_string=db_url,
        source_table="cortex_responses",
        column_map={"query": "prompt", "response": "completion", "context": "context_json"},
        result_table="scroot_scores",
    )

auditor = Auditor()
summary = connector.score_all(auditor)

print(f"Scored {summary['total_scored']} Cortex responses")
print(f"Mean IQS: {summary['mean_iqs']:.3f}")
print(f"Flag counts: {summary['flag_counts']}")
