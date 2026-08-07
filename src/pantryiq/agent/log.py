"""The query log — every question, what was retrieved for it, and the guardrail's verdict.

The design doc calls this `gold.agent_query_log`. **It lives in its own file instead**, and the
deviation is stated rather than silent: `gold/publish.py` swaps Gold wholesale inside a
transaction and the serving copy is opened read-only, so a write-heavy table cannot live there
without either blocking the pipeline or being destroyed by it. `data/agent_log.duckdb` is owned
by the agent, untouched by Gold rebuilds, and gitignored.

What it is for, in order: reproducing a bad answer (the context is stored, so the guardrail can
be re-run against it offline), and tracking the guardrail's live catch rate against the measured
one from `scripts/measure_guardrail.py`.

**The stored context is the whole point.** A log holding only the question and the answer cannot
answer "was this number real?" six weeks later — the retrieved facts are what make a past answer
checkable, so they are stored as JSON alongside it.

Run:  uv run python -m pantryiq.agent.log
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pantryiq.agent.context import AnswerContext
from pantryiq.agent.guardrail import Verdict

# `PANTRYIQ_LOG_DB` overrides the location. This is the one file serving *writes*, so a deployed
# container has to put it on a mount that survives a restart — otherwise the stored contexts, and
# with them the ability to re-check a past answer, vanish on every deploy. Read at import time; see
# the note on `retrieval.DEFAULT_DB`.
DEFAULT_DB = Path(os.environ.get("PANTRYIQ_LOG_DB") or "data/agent_log.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_query_log (
    query_id               VARCHAR PRIMARY KEY,
    asked_at               TIMESTAMPTZ NOT NULL,
    user_input             VARCHAR NOT NULL,
    parsed_filter          VARCHAR NOT NULL,   -- JSON
    retrieved_recipe_ids   VARCHAR[] NOT NULL,
    retrieved_context      VARCHAR NOT NULL,   -- JSON: what the answer was allowed to say
    generated_response     VARCHAR,
    guardrail_pass         BOOLEAN NOT NULL,
    numeric_claims_checked INTEGER NOT NULL,
    unsupported_numbers    DOUBLE[] NOT NULL,
    regenerated            BOOLEAN NOT NULL,
    latency_ms             DOUBLE
)
"""


def record(context: AnswerContext, response: str, verdict: Verdict, regenerated: bool,
           latency_ms: float | None = None, db_path: Path | str = DEFAULT_DB) -> str:
    """Write one row and return its query_id."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    payload = context.to_dict()
    query_id = str(uuid.uuid4())

    con = duckdb.connect(str(db_path))
    try:
        con.execute(SCHEMA)
        con.execute(
            "INSERT INTO agent_query_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [query_id, datetime.now(timezone.utc), context.question,
             json.dumps(payload["query"]),
             [recipe.recipe_id for recipe in context.recipes],
             json.dumps(payload), response,
             verdict.passed, verdict.checked, list(verdict.unsupported),
             regenerated, latency_ms],
        )
    finally:
        con.close()
    return query_id


def summary(db_path: Path | str = DEFAULT_DB) -> dict:
    """The running guardrail rate, to be compared against the measured one."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {"queries": 0}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("""
            SELECT count(*), count(*) FILTER (WHERE guardrail_pass),
                   count(*) FILTER (WHERE regenerated), sum(numeric_claims_checked),
                   median(latency_ms)
            FROM agent_query_log
        """).fetchone()
    finally:
        con.close()
    return {"queries": row[0], "passed": row[1], "regenerated": row[2],
            "claims_checked": row[3] or 0, "median_latency_ms": row[4]}


def main() -> None:
    stats = summary()
    if not stats["queries"]:
        print(f"no queries logged yet — run  uv run python -m pantryiq.agent  ({DEFAULT_DB})")
        return
    print(f"agent_query_log ({DEFAULT_DB}): {stats['queries']:,} queries")
    print(f"  guardrail passed first time : {stats['passed']:,} "
          f"({100 * stats['passed'] / stats['queries']:.1f}%)")
    print(f"  regenerated                 : {stats['regenerated']:,}")
    print(f"  numeric claims checked      : {stats['claims_checked']:,}")
    if stats["median_latency_ms"]:
        print(f"  median latency              : {stats['median_latency_ms']:,.0f} ms")


if __name__ == "__main__":
    main()
