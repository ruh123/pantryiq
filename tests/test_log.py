"""The query log — what has to survive so a past answer stays checkable."""
import json

import duckdb

from pantryiq.agent.context import AnswerContext, RecipeFact
from pantryiq.agent.guardrail import Verdict, check
from pantryiq.agent.log import record, summary
from pantryiq.agent.retrieval import PantryQuery
from test_context import make_candidate


def a_context(question: str = "what can I make?") -> AnswerContext:
    return AnswerContext(question=question, query=PantryQuery(pantry=("chicken",)),
                         recipes=(RecipeFact.from_candidate(make_candidate()),))


def test_one_query_writes_exactly_one_row_with_the_verdict(tmp_path):
    db = tmp_path / "log.duckdb"
    context = a_context()
    verdict = check("about 3,470 kcal", context)

    query_id = record(context, "about 3,470 kcal", verdict, regenerated=False,
                      latency_ms=1234.5, db_path=db)

    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute("SELECT query_id, guardrail_pass, numeric_claims_checked, "
                           "regenerated, latency_ms FROM agent_query_log").fetchall()
    finally:
        con.close()

    assert rows == [(query_id, True, 1, False, 1234.5)]


def test_a_failed_verdict_records_the_offending_numbers(tmp_path):
    """A log that only says "failed" cannot tell you what the model made up."""
    db = tmp_path / "log.duckdb"

    record(a_context(), "about 9,999 kcal", Verdict(False, 1, (9999.0,)),
           regenerated=True, db_path=db)

    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute("SELECT guardrail_pass, unsupported_numbers, regenerated "
                          "FROM agent_query_log").fetchone()
    finally:
        con.close()

    assert row == (False, [9999.0], True)


def test_the_retrieved_context_is_stored_so_the_check_can_be_replayed(tmp_path):
    """A log holding only the question and the answer cannot settle "was that number real?"
    six weeks later. The retrieved facts are what make a past answer checkable."""
    db = tmp_path / "log.duckdb"
    context = a_context()

    record(context, "about 3,470 kcal", check("about 3,470 kcal", context),
           regenerated=False, db_path=db)

    con = duckdb.connect(str(db), read_only=True)
    try:
        stored, ids = con.execute(
            "SELECT retrieved_context, retrieved_recipe_ids FROM agent_query_log").fetchone()
    finally:
        con.close()

    assert ids == ["r1"]
    assert json.loads(stored)["recipes"][0]["total_kcal"] == 3470


def test_repeated_queries_accumulate_rather_than_replacing(tmp_path):
    db = tmp_path / "log.duckdb"
    context = a_context()
    passing = check("about 3,470 kcal", context)

    for _ in range(3):
        record(context, "about 3,470 kcal", passing, regenerated=False, db_path=db)

    stats = summary(db)

    assert stats["queries"] == 3
    assert stats["passed"] == 3


def test_summary_on_a_missing_log_is_empty_rather_than_an_error(tmp_path):
    """The first run of the agent has no log yet, and that is not a failure."""
    assert summary(tmp_path / "absent.duckdb") == {"queries": 0}
