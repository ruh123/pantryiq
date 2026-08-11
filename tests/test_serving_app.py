"""The web app — what has to be on the screen, and what must never be.

Runs against `streamlit.testing.v1.AppTest`, which executes the real script in-process, so the
stubs go on the modules the script imports from. No corpus and no network: the startup guard and
the façade are both substituted, which is also what lets these run in CI.

What is pinned here is the trust surface, not the layout. A recipe total that appears without its
coverage, or a fallback that reads like a normal answer, is the failure this whole project exists
to prevent — a nicer arrangement of the same facts is not.
"""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from pantryiq.agent.guardrail import Verdict
from pantryiq.serving import published
from pantryiq.serving.answer import Answered
from pantryiq.serving.app import MAX_QUESTIONS_PER_SESSION, md, money
from test_context import context_for, make_candidate

APP = "src/pantryiq/serving/app.py"


def an_answer(text: str = "About 3,470 kcal for the whole dish.", **overrides) -> Answered:
    fields = dict(
        question="what can I make?", text=text,
        verdict=Verdict(passed=True, checked=7, unsupported=()),
        context=context_for(make_candidate()),
        regenerated=False, fell_back=False, latency_ms=8_400.0,
        timings=(("parsing", 3_600.0), ("retrieving", 20.0), ("answering", 4_780.0)),
        query_id="q1", log_error=None,
    )
    fields.update(overrides)
    return Answered(**fields)


@pytest.fixture
def app(monkeypatch):
    """A healthy app whose façade is a stub. Mutate `state` to shape what comes back."""
    state = {"answer": an_answer(), "asked": []}

    monkeypatch.setattr("pantryiq.serving.answer.startup_problem", lambda: None)
    monkeypatch.setattr("pantryiq.agent.claude.get_client", lambda: SimpleNamespace())
    monkeypatch.setattr("pantryiq.serving.app.provenance", lambda: None, raising=False)

    def fake_answer(question, client=None, on_stage=None):
        state["asked"].append(question)
        for stage in ("parsing", "retrieving", "answering"):
            if on_stage:
                on_stage(stage)
        return state["answer"]

    monkeypatch.setattr("pantryiq.serving.answer.answer_question", fake_answer)
    return state


def ask(at: AppTest, question: str = "what can I make with chicken?") -> AppTest:
    """Fill the question box and submit the form."""
    at.text_input[0].input(question)
    at.button[0].click()
    return at.run()


def text_of(at: AppTest) -> str:
    """Everything the page rendered, flattened — order and widget type are not the point here.

    Expander *labels* are included deliberately: several caveat titles carry the whole warning
    ("Prices are a stand-in, not real pricing") while the body only elaborates, so a check that
    reads element values alone would miss them and report a clean page.
    """
    parts = [str(element.label) for element in at.expander]
    for kind in ("markdown", "caption", "success", "warning", "error", "info", "text", "header",
                 "subheader", "title"):
        parts.extend(str(element.value) for element in getattr(at, kind))
    return "\n".join(parts)


# --- the startup guard --------------------------------------------------------------------------


def test_a_missing_gold_file_is_explained_rather_than_crashed(monkeypatch):
    """Without the guard this surfaces as a DuckDB IOException from inside retrieval, which does
    not read as "the container was started without its data"."""
    monkeypatch.setattr("pantryiq.serving.answer.startup_problem",
                        lambda: "The Gold data file is missing: data/pantryiq_gold.duckdb")

    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert "Gold data file is missing" in at.error[0].value
    assert not at.tabs


def test_a_healthy_app_shows_the_three_tabs(app):
    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Ask", "Plan a week", "How this works"]


# --- the answer, and the evidence beside it -------------------------------------------------------


def test_the_answer_and_its_ledger_are_both_rendered(app):
    at = ask(AppTest.from_file(APP).run())
    page = text_of(at)

    assert not at.exception
    assert "About 3,470 kcal for the whole dish." in page
    assert "What the answer was allowed to say" in page
    assert "Chicken Bake" in page


def test_a_recipe_total_never_appears_without_its_coverage(app):
    """`context.py`: "Coverage is a required field, never optional." A bare total is a stronger
    claim than the warehouse supports."""
    at = ask(AppTest.from_file(APP).run())
    page = text_of(at)

    assert "3,470 kcal for the whole dish" in page
    assert "4 of 5 ingredients weighed" in page
    assert "undercount" in page


def test_a_partly_priced_recipe_is_shown_as_a_floor(app):
    """The fixture recipe is 40% priced. "$11.05" would be a claim about the dish nobody made."""
    at = ask(AppTest.from_file(APP).run())

    assert "at least $11.05" in text_of(at)


def test_the_trust_score_is_on_the_screen(app):
    """§4 of the brief asks for this by name — the demo-able quality signal."""
    at = ask(AppTest.from_file(APP).run())
    page = text_of(at)

    assert "data_trust_score" in page
    assert "0.56" in page          # the fixture recipe's score, rendered off RecipeFact unchanged
    assert "nutrition coverage" in page


def test_the_stage_timings_are_shown(app):
    """Retrieval in tens of ms against seconds per model call is the architecture in one line."""
    at = ask(AppTest.from_file(APP).run())
    page = text_of(at)

    assert "20 ms" in page and "3,600 ms" in page
    assert "8,400 ms" in page


# --- the badge: four states, because `regenerated` alone conflates two of them ---------------------


def test_a_clean_answer_says_what_was_checked(app):
    at = ask(AppTest.from_file(APP).run())

    assert "Guardrail PASS" in at.success[0].value
    assert "7 numeric claims" in at.success[0].value


def test_a_failed_check_names_the_numbers_that_trace_to_nothing(app):
    app["answer"] = an_answer(verdict=Verdict(passed=False, checked=7, unsupported=(2900.0,)))

    at = ask(AppTest.from_file(APP).run())

    assert "Guardrail FAIL" in at.error[0].value
    assert "2,900" in at.error[0].value


def test_a_regenerated_answer_is_flagged_without_being_called_a_failure(app):
    app["answer"] = an_answer(regenerated=True)

    at = ask(AppTest.from_file(APP).run())

    assert "rejected the first draft" in at.warning[0].value
    assert "passed" in at.warning[0].value


def test_a_fallback_is_not_dressed_up_as_an_answer(app):
    """The whole point of `fell_back`. Both drafts were rejected; presenting the templated facts
    as though Claude wrote them would hide the one failure the user most needs to know about."""
    app["answer"] = an_answer(regenerated=True, fell_back=True)

    at = ask(AppTest.from_file(APP).run())

    assert "rejected two drafts" in at.warning[0].value


def test_an_unlogged_answer_says_so_rather_than_pretending_it_was_logged(app):
    app["answer"] = an_answer(query_id=None, log_error="OSError: database is locked")

    at = ask(AppTest.from_file(APP).run())

    assert "Not written to the query log" in text_of(at)


def test_an_empty_retrieval_is_stated_rather_than_left_blank(app):
    """The "unobtainium and moon cheese" case: nothing matched, and the screen has to say that
    rather than show an answer floating above an empty panel."""
    empty = context_for(make_candidate())
    app["answer"] = an_answer(text="Nothing in the warehouse matches that.",
                              context=type(empty)(question=empty.question, query=empty.query,
                                                  recipes=()))

    at = ask(AppTest.from_file(APP).run())

    assert "Retrieval returned nothing" in text_of(at)


# --- the standing caveats -------------------------------------------------------------------------


def test_every_caveat_is_on_the_page_whether_or_not_the_prose_mentions_it(app):
    """These are structural limits a fluent answer can hide, so they are not left to the model."""
    at = AppTest.from_file(APP).run()
    page = text_of(at)

    for title, _ in published.CAVEATS:
        assert title in page, f"missing caveat: {title}"


def test_the_stand_in_pricing_and_the_truncated_corpus_are_both_disclosed(app):
    at = AppTest.from_file(APP).run()
    page = text_of(at)

    assert "stand-in" in page
    assert "Pickled Bologna" in page


# --- the abuse cap ------------------------------------------------------------------------------


def test_a_session_cannot_ask_forever(app):
    """A public URL in front of a metered key.

    The cap is exercised for real rather than lowered: `AppTest` execs the script into a fresh
    namespace on every run, so a module-level constant cannot be monkeypatched — the substituted
    façade makes the extra runs free anyway.
    """
    at = AppTest.from_file(APP).run()

    for _ in range(MAX_QUESTIONS_PER_SESSION + 2):
        ask(at)

    assert len(app["asked"]) == MAX_QUESTIONS_PER_SESSION
    assert any("questions per session" in element.value for element in at.warning)


# --- the floor rule, tested directly ---------------------------------------------------------------


def test_two_costs_in_one_paragraph_survive_the_markdown_renderer(app):
    """Streamlit reads `$...$` as inline LaTeX. An answer saying "comes to $33.65 total, leaving
    $16.35 of your $50 budget" rendered the middle as a serif formula and the amounts vanished.
    Cost is the figure this app quotes most, so the pairing is a live hazard on every tab."""
    app["answer"] = an_answer(text="Comes to $33.65 total, leaving $16.35 of your $50 budget.")

    at = ask(AppTest.from_file(APP).run())
    page = text_of(at)

    for amount in ("33.65", "16.35", "50"):
        assert amount in page
    assert "\\$33.65" in page, "dollar signs reached the renderer unescaped"


def test_the_escape_is_display_only_and_changes_no_number():
    """The guardrail checks the unescaped text, so escaping must not alter a value."""
    assert md("at least $11.05") == "at least \\$11.05"
    assert md("1,968 kcal") == "1,968 kcal"


@pytest.mark.parametrize(("value", "coverage", "expected"), [
    (11.0512, 1.0, "$11.05"),
    (11.0512, 0.4, "at least $11.05"),
    (11.0512, None, "$11.05"),
    (None, 1.0, "not priced"),
])
def test_the_cost_floor_rule(value, coverage, expected):
    assert money(value, coverage) == expected
