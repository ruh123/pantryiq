"""What the web app calls — the agent's four stages, with everything the screen has to show.

`agent/__main__.py:ask()` returns a bare string. It *computes* the verdict, the retrieved facts,
whether the answer was regenerated and how long it took, and then discards all four — which is
precisely the set the brief asks the UI to surface inline (§4: "surfaces verified numbers +
`data_trust_score`"). `main()` in that module keeps them, prints them, and throws them away at the
end of the loop. This is that loop with a return type.

Three things here that the CLI path does not do, each because a browser is not a terminal:

**Logging is best-effort.** `log.record` opens a read-WRITE DuckDB connection, and DuckDB's file
lock is held across processes. A deploy with two containers, or a volume that filled up, must not
destroy an answer that already passed the guardrail — the answer is the product, the log row is
the audit trail. A failure lands in `log_error` and the answer is returned anyway.

**A fallback is distinguishable from a retry.** `guarded_answer` returns `regenerated=True` for
both "the second draft passed" and "both drafts failed, here are the facts instead". Those are
very different things to show a user, and the tuple cannot tell them apart, so `fell_back`
recovers it by comparing against `templated_answer`, which is deterministic.

**Stages are reported as they start**, so the 7-12 second wait has something honest in it. There
are three, not four: `guarded_answer` owns generate -> check -> regenerate -> check -> fall back
as one unit, and the façade cannot see that boundary without duplicating the ladder. A "checking"
stage that always measured 0 ms would be instrumentation that describes the code it wishes existed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from pantryiq.agent.claude import Refused, get_client, require_api_key
from pantryiq.agent.context import AnswerContext, build
from pantryiq.agent.guardrail import Verdict, guarded_answer, templated_answer
from pantryiq.agent.log import record
from pantryiq.agent.parse import parse
from pantryiq.agent.planner import Plan, build_plan, plan_context
from pantryiq.agent.retrieval import DEFAULT_DB as GOLD_DB

# Reported through `on_stage` as each begins. Named here so the UI's labels and the façade's
# calls cannot drift apart.
STAGES = ("parsing", "retrieving", "answering")
PLAN_STAGES = ("selecting", "narrating")


class ServingError(RuntimeError):
    """The question could not be answered, phrased for someone who is not reading a traceback."""


@dataclass(frozen=True)
class Answered:
    """One answered question, and the evidence for judging the answer."""

    question: str
    text: str
    verdict: Verdict
    context: AnswerContext
    regenerated: bool
    fell_back: bool
    latency_ms: float
    # (stage, milliseconds) in order. Retrieval runs in tens of milliseconds against several
    # seconds for each model call, which is the architecture made visible: the part that decides
    # what may be said is the cheap part.
    timings: tuple[tuple[str, float], ...]
    query_id: str | None
    log_error: str | None


@dataclass(frozen=True)
class PlannedWeek:
    """A plan chosen in code, the constraint check on it, and Claude's description of it."""

    plan: Plan
    violations: tuple[str, ...]
    narration: Answered


def _durations(marks: list[tuple[str, float]], end: float) -> tuple[tuple[str, float], ...]:
    """Consecutive marks into (stage, ms) spans."""
    edges = [clock for _, clock in marks] + [end]
    return tuple((name, (edges[index + 1] - edges[index]) * 1000)
                 for index, (name, _) in enumerate(marks))


def startup_problem() -> str | None:
    """Why this process cannot answer anything, phrased for a page, or None if it can.

    Checked once when the app loads rather than discovered on the first question. Both failures
    are configuration, and both are silent until something tries to use them: a missing Gold file
    surfaces as a DuckDB IOException from inside retrieval, and a missing key as a SystemExit from
    inside the SDK's header builder. Neither reads as "the container was started wrong".
    """
    if not GOLD_DB.exists():
        return (
            f"The Gold data file is missing: {GOLD_DB}\n\n"
            "It is a build artifact, not part of the repository — the recipe corpus it is "
            "derived from is redistributed under terms that do not permit publishing it. Build "
            "it with `make gold`, or point PANTRYIQ_GOLD_DB at an existing copy."
        )
    try:
        require_api_key()
    except SystemExit as exc:
        return str(exc)
    return None


def _client(client):
    """A client, or a renderable error.

    Built before the clock starts, and deliberately: a web session builds one and passes it to
    every question, so reading `.env` and constructing the SDK object is not time the user spent
    waiting for an answer. `require_api_key` raises SystemExit — a BaseException, which an
    `except Exception` at the UI boundary would let straight through.
    """
    if client is not None:
        return client
    try:
        return get_client()
    except SystemExit as exc:
        raise ServingError(str(exc)) from exc


def _log(context: AnswerContext, text: str, verdict: Verdict, regenerated: bool,
         latency_ms: float) -> tuple[str | None, str | None]:
    """Write the audit row, or report why it could not be written. Never raises.

    The broad except is deliberate and narrow in effect: everything above this point has already
    succeeded, so the only question left is whether the caller also gets a `query_id`.
    """
    try:
        return record(context, text, verdict, regenerated, latency_ms), None
    except Exception as exc:  # noqa: BLE001 - see the docstring
        return None, f"{type(exc).__name__}: {exc}"


def answer_question(question: str, client=None, on_stage=None) -> Answered:
    """Parse, retrieve, answer under the guardrail, log. Raises `ServingError` if it cannot."""
    marks: list[tuple[str, float]] = []

    def stage(name: str) -> None:
        marks.append((name, time.perf_counter()))
        if on_stage is not None:
            on_stage(name)

    client = _client(client)
    try:
        stage("parsing")
        query = parse(question, client)
        stage("retrieving")
        context = build(question, query)
        stage("answering")
        text, verdict, regenerated = guarded_answer(context, client)
    except Refused as exc:
        # A 200 with an empty body — see `claude.py`. Nothing downstream catches this today.
        raise ServingError(
            "Claude declined to handle that request. Rephrasing it usually clears it."
        ) from exc

    finished = time.perf_counter()
    # The first mark IS the start, so the spans tile the interval instead of sampling it.
    latency_ms = (finished - marks[0][1]) * 1000
    query_id, log_error = _log(context, text, verdict, regenerated, latency_ms)

    return Answered(
        question=question, text=text, verdict=verdict, context=context,
        regenerated=regenerated,
        fell_back=regenerated and text == templated_answer(context),
        latency_ms=latency_ms, timings=_durations(marks, finished),
        query_id=query_id, log_error=log_error,
    )


def plan_week(budget_usd: float, kcal_per_day: float, days: int,
              client=None, on_stage=None) -> PlannedWeek:
    """Solve the week in code, verify it, then let Claude describe what was already decided.

    Uses `guarded_answer` rather than `planner.main()`'s raw `answer` + `check`: the CLI has a
    human reading the FAIL line, a web page does not, so the narration gets the same
    regenerate-once-then-fall-back ladder every other answer gets.
    """
    marks: list[tuple[str, float]] = []

    def stage(name: str) -> None:
        marks.append((name, time.perf_counter()))
        if on_stage is not None:
            on_stage(name)

    client = _client(client)
    try:
        stage("selecting")
        plan = build_plan(budget_usd=budget_usd, kcal_per_day=kcal_per_day, days=days)
        # Checked before the model sees any of it: a narration of an invalid plan is a fluent
        # description of something wrong, which is worse than no narration.
        violations = tuple(plan.violations())
        context = plan_context(plan)
        stage("narrating")
        text, verdict, regenerated = guarded_answer(context, client)
    except Refused as exc:
        raise ServingError(
            "Claude declined to describe this plan. The plan itself is unaffected — it was "
            "chosen and checked in code."
        ) from exc

    finished = time.perf_counter()
    latency_ms = (finished - marks[0][1]) * 1000
    query_id, log_error = _log(context, text, verdict, regenerated, latency_ms)

    narration = Answered(
        question=context.question, text=text, verdict=verdict, context=context,
        regenerated=regenerated,
        fell_back=regenerated and text == templated_answer(context),
        latency_ms=latency_ms, timings=_durations(marks, finished),
        query_id=query_id, log_error=log_error,
    )
    return PlannedWeek(plan=plan, violations=violations, narration=narration)
