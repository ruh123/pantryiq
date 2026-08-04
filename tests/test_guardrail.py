"""The numeric guardrail — what it must catch, and what it must not block.

Both directions are load-bearing. A guardrail that rejects everything catches every fabrication
and is useless, so the tests that assert a *correct* answer passes matter as much as the ones
that assert a wrong one fails.
"""
import pytest

from pantryiq.agent.context import AnswerContext, RecipeFact
from pantryiq.agent.guardrail import (
    check,
    extract_numbers,
    permitted,
    templated_answer,
)
from pantryiq.agent.retrieval import PantryQuery
from test_context import make_candidate


def context_of(*candidates) -> AnswerContext:
    return AnswerContext(question="q", query=PantryQuery(pantry=("chicken", "rice")),
                         recipes=tuple(RecipeFact.from_candidate(c)
                                       for c in (candidates or (make_candidate(),))))


# ------------------------------------------------------------- extraction

@pytest.mark.parametrize("text,expected", [
    ("about 3,470 kcal", [3470.0]),
    ("costs $11.05 today", [11.05]),
    ("80% of ingredients", [80.0]),
    ("between 1,200-1,500 kcal", [1200.0, 1500.0]),      # en-dash and hyphen ranges split
    ("between 1,200–1,500 kcal", [1200.0, 1500.0]),
    ("4 of 5 weighed.", [4.0, 5.0]),                      # a trailing full stop is not a decimal
    ("half a cup", []),
])
def test_numbers_are_extracted_from_ordinary_prose(text, expected):
    assert extract_numbers(text) == expected


def test_a_recipe_id_is_an_identifier_not_a_quantity():
    """Both false positives in the first measured run were recipe ids. A model citing
    `recipenlg:14797` is quoting the context exactly, and reading the digits inside an identifier
    as a figure blocked two correct answers."""
    text = "Shrimp Scampi (recipenlg:14797) is about 3,470 kcal."

    assert extract_numbers(text) == [14797.0, 3470.0]
    assert extract_numbers(text, ignore=["recipenlg:14797"]) == [3470.0]
    assert check(text, context_of(make_candidate(recipe_id="recipenlg:14797"))).passed


# ------------------------------------------------------------- what must fail

def test_an_invented_figure_is_caught():
    verdict = check("It is about 2,900 kcal for the whole dish.", context_of())

    assert not verdict.passed
    assert verdict.unsupported == (2900.0,)


def test_a_total_divided_by_its_serving_count_is_caught():
    """The number §3.4 refused to publish. `servings` is 8 and `total_kcal` is 3,470, both real;
    3470/8 = 433.75 is not, because coverage is 0.8 and the total is a sum over four fifths of
    the ingredients. Admitting division as a derivation would sanction exactly that."""
    verdict = check("That is roughly 434 calories per serving.", context_of())

    assert not verdict.passed
    assert verdict.unsupported == (434.0,)


def test_a_small_fabricated_count_is_caught_when_a_unit_follows_it():
    """A bare small integer has to be tolerated as a list marker ("1.", "2."). That allowance
    once left the most consequential small claim unguarded: an invented serving count. The unit
    after the number is what separates enumeration from measurement."""
    assert permitted(3.0, allowed={100.0}, ordinal_limit=5, quantified=False), "a list marker"
    assert not permitted(3.0, allowed={100.0}, ordinal_limit=5, quantified=True), "a claim"

    # End to end: 3 is not a value in this context, so only the unit decides it.
    assert not check("It makes about 3 servings.", context_of()).passed


def test_a_count_must_be_exact_rather_than_within_ten_percent():
    """A 10% band around 5 spans 4.5 to 5.5, which would let an answer claim 6 of 5 ingredients
    were weighed. Counts have to be right, not close."""
    assert not check("6 ingredients were weighed.", context_of()).passed


def test_an_invented_cost_is_caught():
    assert not check("You can make it for about $4.25.", context_of()).passed


# ------------------------------------------------------------- what must pass

def test_a_faithful_answer_passes():
    """The control. Everything here is in the context: title, total, coverage counts, cost."""
    text = ("Chicken Bake uses chicken and rice, about 3,470 kcal for the whole dish "
            "(4 of 5 ingredients weighed), and at least $11.05.")

    verdict = check(text, context_of())

    assert verdict.passed, verdict.unsupported
    assert verdict.checked == 4  # 3,470 / 4 / 5 / 11.05


def test_a_rounded_quotation_passes():
    """Models write 'about 3,500' for 3,470. Tolerance reuses er/nutrition.error so 'within 10%'
    means the same thing here as everywhere else in the project."""
    assert check("roughly 3,500 kcal", context_of()).passed


def test_coverage_quoted_as_a_percentage_passes():
    assert check("80% of the ingredients were weighed.", context_of()).passed


def test_the_users_own_constraint_can_be_restated():
    context = AnswerContext(question="q", query=PantryQuery(pantry=("chicken",), max_kcal=500),
                            recipes=(RecipeFact.from_candidate(make_candidate()),))

    assert check("All of these come in under 500 calories.", context).passed


# ------------------------------------------------------------- degradation

def test_the_fallback_answer_is_built_from_the_context_and_passes_its_own_check():
    """A guardrail failure must cost the user a plainer answer, never a wrong one — so the
    templated fallback has to be correct by construction."""
    context = context_of()

    text = templated_answer(context)

    assert check(text, context).passed
    assert "Chicken Bake" in text


def test_the_fallback_for_an_empty_result_does_not_invent_a_recipe():
    empty = AnswerContext(question="q", query=PantryQuery(pantry=("unobtainium",)), recipes=())

    text = templated_answer(empty)

    assert "Nothing in the warehouse matches" in text
    assert check(text, empty).passed


def test_the_complaint_names_the_offending_number_and_forbids_dividing():
    """A regeneration that is not told what was wrong is just a re-roll of the same dice."""
    verdict = check("That is roughly 434 calories per serving.", context_of())

    complaint = verdict.complaint()

    assert "434" in complaint
    assert "divide" in complaint


def test_permitted_rejects_a_value_near_nothing_in_the_allowed_set():
    assert permitted(3470.0, {3470.0}, 0)
    assert not permitted(9999.0, {3470.0}, 0)
