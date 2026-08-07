"""The façade the web app calls — what it keeps, and what it refuses to lose.

`ask()` returns a string; a screen needs the verdict, the retrieved facts and the timings too.
These pin that those survive, that a failed audit write cannot take an answer down with it, and
that the two exceptions nothing in the agent catches arrive as something renderable.

No corpus and no network: every stage is substituted, because what is under test is the wiring
between them, not the stages themselves.
"""
import pytest

from pantryiq.agent.guardrail import Verdict, templated_answer
from pantryiq.serving.answer import (
    PLAN_STAGES,
    STAGES,
    Answered,
    ServingError,
    answer_question,
    plan_week,
)
from test_context import context_for, make_candidate

MODULE = "pantryiq.serving.answer"


@pytest.fixture
def wired(monkeypatch):
    """Substitute every stage. Returns a dict the test can mutate to shape the run."""
    context = context_for(make_candidate())
    state = {
        "context": context,
        "text": "about 3,470 kcal (4 of 5 ingredients weighed).",
        "verdict": Verdict(passed=True, checked=3, unsupported=()),
        "regenerated": False,
        "recorded": [],
        "stages": [],
    }

    monkeypatch.setattr(f"{MODULE}.get_client", lambda: object())
    monkeypatch.setattr(f"{MODULE}.parse", lambda question, client: context.query)
    monkeypatch.setattr(f"{MODULE}.build", lambda question, query: state["context"])
    monkeypatch.setattr(
        f"{MODULE}.guarded_answer",
        lambda ctx, client: (state["text"], state["verdict"], state["regenerated"]),
    )

    def fake_record(ctx, text, verdict, regenerated, latency_ms):
        state["recorded"].append((ctx, text, verdict, regenerated, latency_ms))
        return "query-1"

    monkeypatch.setattr(f"{MODULE}.record", fake_record)
    return state


def test_it_returns_the_four_things_ask_throws_away(wired):
    """`ask()` computes the verdict, the facts, the regenerate flag and the latency, then returns
    only the string. Every one of them is something the screen has to show."""
    result = answer_question("what can I make?", client=object())

    assert isinstance(result, Answered)
    assert result.verdict.checked == 3 and result.verdict.passed
    assert result.context.recipes[0].data_trust_score == 0.56
    assert result.regenerated is False
    assert result.latency_ms > 0
    assert result.query_id == "query-1"


def test_stages_are_reported_in_order_as_each_begins(wired):
    """The wait is 7-12 seconds with nothing streaming, so the stage is the only honest signal
    the screen has. Reported on entry, not on completion."""
    seen = []

    answer_question("q", client=object(), on_stage=seen.append)

    assert tuple(seen) == STAGES


def test_the_timings_cover_the_whole_run_and_name_the_same_stages(wired):
    """Retrieval against the model calls is the architecture made visible, so the spans have to
    add up rather than sample."""
    result = answer_question("q", client=object())

    assert tuple(name for name, _ in result.timings) == STAGES
    assert sum(ms for _, ms in result.timings) == pytest.approx(result.latency_ms, rel=0.05)


def test_a_failed_audit_write_does_not_take_the_answer_with_it(wired, monkeypatch):
    """The answer already passed the guardrail. DuckDB's write lock is held across processes, so
    a second container or a full volume makes this a live failure mode, not a hypothetical."""
    def exploding_record(*args, **kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(f"{MODULE}.record", exploding_record)

    result = answer_question("q", client=object())

    assert result.text.startswith("about 3,470 kcal")
    assert result.query_id is None
    assert result.log_error == "OSError: database is locked"


def test_a_regenerated_answer_is_not_reported_as_a_fallback(wired):
    """Draft two passing is a good outcome. Saying "we gave up" there would be a lie."""
    wired["regenerated"] = True

    result = answer_question("q", client=object())

    assert result.regenerated is True
    assert result.fell_back is False


def test_the_templated_fallback_is_distinguishable_from_a_retry(wired):
    """`guarded_answer` returns `regenerated=True` for both, so the tuple alone cannot tell a
    user whether they are reading Claude's second attempt or the raw facts."""
    wired["text"] = templated_answer(wired["context"])
    wired["regenerated"] = True

    result = answer_question("q", client=object())

    assert result.fell_back is True


def test_a_refusal_arrives_as_something_a_page_can_render(wired, monkeypatch):
    """`Refused` propagates uncaught through parse, answer and guarded_answer today."""
    from pantryiq.agent.claude import Refused

    def refusing_parse(question, client):
        raise Refused("stop_reason='refusal'")

    monkeypatch.setattr(f"{MODULE}.parse", refusing_parse)

    with pytest.raises(ServingError, match="declined"):
        answer_question("q", client=object())


def test_a_missing_api_key_does_not_end_the_process(wired, monkeypatch):
    """`require_api_key` raises SystemExit, which is a BaseException: an `except Exception` at the
    UI boundary would miss it and the script run would end with no explanation."""
    def keyless():
        raise SystemExit("ANTHROPIC_API_KEY is unset or empty.")

    monkeypatch.setattr(f"{MODULE}.get_client", keyless)

    with pytest.raises(ServingError, match="ANTHROPIC_API_KEY"):
        answer_question("q", client=None)


def test_the_client_is_only_built_when_one_is_not_supplied(wired, monkeypatch):
    """A web session builds the client once; rebuilding it per question re-reads .env every time."""
    monkeypatch.setattr(f"{MODULE}.get_client",
                        lambda: pytest.fail("get_client called despite a supplied client"))

    answer_question("q", client=object())


# --- the week planner -------------------------------------------------------------------------


@pytest.fixture
def wired_plan(monkeypatch, wired):
    from pantryiq.agent.planner import Plan

    plan = Plan(recipes=(make_candidate(cost_total_usd=6.0, total_kcal=1800.0),),
                budget_usd=50.0, kcal_per_day=2000.0, days=7)
    monkeypatch.setattr(f"{MODULE}.build_plan",
                        lambda budget_usd, kcal_per_day, days: plan)
    monkeypatch.setattr(f"{MODULE}.plan_context", lambda p: wired["context"])
    return plan


def test_the_plan_is_verified_in_code_and_the_result_travels_with_it(wired, wired_plan):
    """`Plan.violations()` is the check; a narration of an invalid plan is a fluent description of
    something wrong."""
    result = plan_week(50.0, 2000.0, 7, client=object())

    assert result.plan is wired_plan
    assert result.violations == ()
    assert result.narration.verdict.passed


def test_a_plan_that_breaks_its_budget_reports_it_rather_than_narrating_over_it(
    wired, wired_plan, monkeypatch
):
    from pantryiq.agent.planner import Plan

    over = Plan(recipes=(make_candidate(cost_total_usd=90.0, total_kcal=1800.0),),
                budget_usd=50.0, kcal_per_day=2000.0, days=7)
    monkeypatch.setattr(f"{MODULE}.build_plan", lambda budget_usd, kcal_per_day, days: over)

    result = plan_week(50.0, 2000.0, 7, client=object())

    assert any("exceeds" in problem for problem in result.violations)


def test_the_narration_goes_through_the_guardrail_ladder(wired, wired_plan):
    """`planner.main()` uses raw `answer` + `check` — no regenerate, no fallback. A CLI has a
    human reading the FAIL line; a web page does not."""
    wired["text"] = templated_answer(wired["context"])
    wired["regenerated"] = True

    result = plan_week(50.0, 2000.0, 7, client=object())

    assert result.narration.fell_back is True


def test_plan_stages_are_reported_too(wired, wired_plan):
    seen = []

    plan_week(50.0, 2000.0, 7, client=object(), on_stage=seen.append)

    assert tuple(seen) == PLAN_STAGES
