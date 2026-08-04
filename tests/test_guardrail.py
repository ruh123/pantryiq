"""The numeric guardrail — what it must catch, and what it must not block.

Both directions are load-bearing. A guardrail that rejects everything catches every fabrication
and is useless, so the tests that assert a *correct* answer passes matter as much as the ones
that assert a wrong one fails.
"""
import pytest

from pantryiq.agent.context import AnswerContext, RecipeFact
from pantryiq.agent.guardrail import (
    mentions,
    guarded_answer,
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


def test_a_bare_integer_passes_only_where_it_is_actually_a_list_marker():
    """The old rule allowed any small integer unless a word from a fixed unit list followed it —
    the same denylist anti-pattern this project condemns in the dietary tags. It let through
    "It serves 2", "Makes 2 portions" and "$2 total". Position is not a vocabulary."""
    assert permitted(3.0, exact=set(), measured={100.0}, is_marker=True), "a list marker"
    assert not permitted(3.0, exact=set(), measured={100.0}, is_marker=False), "a claim"

    assert check("3. Chicken Bake is a good option.", context_of()).passed
    for phrasing in ("It makes about 3 servings.", "It serves 3.", "Makes 3 portions.",
                     "Cut it into 3 pieces.", "About $3 total."):
        assert not check(phrasing, context_of()).passed, phrasing


def test_a_count_gets_no_tolerance_at_any_magnitude():
    """Counts and measurements are split by KIND, not by size. The old rule keyed off magnitude
    (`EXACT_BELOW = 20`) and so treated every small measurement as a count.

    The name says "at any magnitude" and an earlier version only asserted 5 vs 6 — where 1% of 5
    is 0.05, so applying the band to the exact set changed nothing and the mutation survived.
    The property only bites above 100, so that is where it is now asserted.
    """
    assert not permitted(6.0, exact={5.0}, measured=set()), "6 of 5 ingredients"
    # 1005 is inside a 1% band around 1000. As a COUNT it must still be refused.
    assert not permitted(1005.0, exact={1000.0}, measured=set()), "a count, near but not equal"
    assert permitted(1005.0, exact=set(), measured={1000.0}), "the same gap, as a measurement"
    assert permitted(13.0, exact=set(), measured={13.1}), "about 13 g of protein"


def test_a_rounded_small_measurement_is_not_rejected():
    """The regression that made this split necessary: `EXACT_BELOW = 20` rejected "about 13 g of
    protein" against a stored 13.1, measured at 51% of rounded sub-20g macro quotations. The
    20-question measurement suite could not see it because those answers quote total_kcal."""
    small = context_of(make_candidate(protein_g=13.1, fat_g=4.2))

    assert check("Chicken Bake has about 13 g of protein and 4 g of fat.", small).passed


def test_permitted_rejects_a_value_near_nothing_in_the_scope():
    """The rejected value sits just OUTSIDE the band, not far from it. 9,999 against 3,470 stays
    rejected with the tolerance set anywhere up to ~187%, so it pinned nothing."""
    assert permitted(3470.0, exact=set(), measured={3470.0})
    assert permitted(3500.0, exact=set(), measured={3470.0}), "inside 1%"
    assert not permitted(3506.0, exact=set(), measured={3470.0}), "just outside 1%"
    assert not permitted(9999.0, exact={3470.0}, measured=set())


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
    """Models write 'about 3,500' for 3,470. `QUOTE_TOLERANCE` is this module's own 1%, not
    `er/nutrition.EQUIVALENT` — that constant means "two USDA foods are nutritionally
    interchangeable", and reusing it as quoting accuracy accepted 79% of the number line."""
    assert check("roughly 3,500 kcal", context_of()).passed
    assert not check("roughly 3,800 kcal", context_of()).passed


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


# ------------------------------------------------------- per-recipe scoping

def two_recipes() -> AnswerContext:
    return context_of(
        make_candidate(recipe_id="r1", title="Favorite Chicken", total_kcal=3470,
                       ingredient_count=5, counted_ingredients=4, nutrition_coverage=0.8),
        make_candidate(recipe_id="r2", title="Broccoli Casserole", total_kcal=1968,
                       ingredient_count=6, counted_ingredients=6, nutrition_coverage=1.0))


def test_one_recipes_figures_may_not_be_attached_to_another():
    """The failure the union scope could not see, and the reason `check` scopes at all. Measured
    at a median 133% error, up to 1,105% — and in 7 of 20 contexts the swap also upgraded an
    incomplete recipe to full coverage, falsifying the project's central honesty guarantee."""
    verdict = check("Favorite Chicken is your best match — about 1,968 kcal for the whole dish "
                    "(6 of 6 ingredients weighed).", two_recipes())

    assert not verdict.passed
    assert set(verdict.unsupported) == {1968.0, 6.0}


def test_each_recipe_may_state_its_own_figures():
    """The control for the test above: the same numbers, attached correctly, must pass."""
    text = ("Favorite Chicken is about 3,470 kcal (4 of 5 ingredients weighed). "
            "Broccoli Casserole is about 1,968 kcal (6 of 6 ingredients weighed).")

    assert check(text, two_recipes()).passed


def test_a_sentence_naming_two_recipes_puts_both_in_scope():
    """Section-level scoping rejected this and it is a correct sentence. Scope is decided per
    sentence so a summary line covering two dishes can quote both of their figures."""
    text = ("Favorite Chicken and Broccoli Casserole are also options, at about 3,470 and "
            "1,968 kcal respectively.")

    assert check(text, two_recipes()).passed


def test_a_title_that_contains_another_title_does_not_split_it():
    """`Potato Casserole` is a substring of `Hash Brown Potato Casserole`. Resolving overlaps by
    position alone cut one recipe's section in two and checked its calories against the other
    recipe's values — a false positive on a correct answer.

    Asserted on `mentions()` directly, because the end-to-end version could not see the bug it
    names: a spurious extra mention only WIDENS a sentence's scope, and a passing assertion can
    never detect a scope that is too wide. It survived three separate mutations to the overlap
    resolution it exists to protect.
    """
    context = context_of(
        make_candidate(recipe_id="rcp-hash", title="Hash Brown Potato Casserole", total_kcal=4459),
        make_candidate(recipe_id="rcp-plain", title="Potato Casserole", total_kcal=6982))
    text = "Hash Brown Potato Casserole is about 4,459 kcal."

    found = mentions(text, context)

    assert [recipe.title for _, recipe in found] == ["Hash Brown Potato Casserole"], \
        "the longer title must claim the span outright"
    assert check(text, context).passed
    # And the negative: the inner recipe's figure may not be attached to the outer one.
    assert not check("Hash Brown Potato Casserole is about 6,982 kcal.", context).passed


def test_a_title_with_a_nested_parenthetical_still_matches():
    """The corpus writes `Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies)`. Stripping a
    trailing `\\(...\\)` cannot handle the nesting, and when the title failed to match, that
    recipe's whole section was attributed to the recipe named before it.

    The earlier version was rescued twice over and so pinned nothing: with no mention found,
    `check` falls back to the whole-context union which still contained the value, and the value
    also sat inside 1% of the fixture's default `total_grams`. This names a PRIOR recipe — the
    failure the docstring actually describes — and asserts the mention is found.
    """
    context = context_of(
        make_candidate(recipe_id="rcp-baked", title="Baked Oatmeal", total_kcal=1364,
                       total_grams=900, kcal_per_100g=151),
        make_candidate(recipe_id="rcp-drops", total_kcal=2004, total_grams=700,
                       kcal_per_100g=286,
                       title="Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies)"))
    text = ("Baked Oatmeal is about 1,364 kcal. "
            "Honey Oatmeal Drop Cookies is about 2,004 kcal.")

    assert [r.title for _, r in mentions(text, context)][1].startswith("Honey Oatmeal")
    assert check(text, context).passed


def test_a_recipe_can_be_matched_by_its_id_as_well_as_its_title():
    """`_needles` offers the recipe_id as a spelling, and both mutations to that branch survived
    — because every fixture id in the suite was two characters, below its own `len >= 3` floor,
    leaving id matching inert everywhere."""
    context = context_of(
        make_candidate(recipe_id="recipenlg:11613", title=None, total_kcal=3470),
        make_candidate(recipe_id="recipenlg:1031", title=None, total_kcal=1968))

    assert not check("recipenlg:11613 is about 1,968 kcal.", context).passed
    assert check("recipenlg:11613 is about 3,470 kcal.", context).passed


def test_digits_inside_a_recipe_title_are_not_read_as_a_claim():
    """Naming that dish puts a 22 and a 2 into the answer. They are its name, not a measurement
    of it — the same class as a recipe id or the column `kcal_per_100g`."""
    context = context_of(make_candidate(
        title="Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies)", total_kcal=2004))

    assert check("Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies) is 2,004 kcal.",
                 context).passed


def test_an_answer_naming_no_recipe_falls_back_to_the_whole_context():
    """Degrading to the union is the right fallback when nothing can be attributed — it is the
    old behaviour, and it is only reached when the model names no dish at all."""
    assert check("Two of these come in under 3,470 kcal.", two_recipes()).passed


# ------------------------------------------------------- the regenerate loop

def test_guarded_answer_returns_the_verdict_that_describes_what_it_returns(monkeypatch):
    """The loop had NO test at all — inverting it left all 520 green. And it was wrong: the
    fallback was returned paired with the *rejected retry's* verdict, so `log.record` stored a
    clean answer alongside guardrail_pass=False and numbers appearing nowhere in it, corrupting
    the running catch rate on exactly the rows where the guardrail had fired."""
    context = context_of()
    monkeypatch.setattr("pantryiq.agent.generate.answer",
                        lambda ctx, client=None, note="", on_text=None: "It is 9,876 kcal.")

    text, verdict, regenerated = guarded_answer(context, client=object())

    assert regenerated
    assert verdict.passed, "the returned verdict must describe the returned text"
    assert check(text, context).passed
    assert "Chicken Bake" in text, "should have degraded to the templated answer"


def test_guarded_answer_returns_a_first_pass_answer_untouched(monkeypatch):
    context = context_of()
    monkeypatch.setattr("pantryiq.agent.generate.answer",
                        lambda ctx, client=None, note="", on_text=None:
                        "Chicken Bake is about 3,470 kcal.")

    text, verdict, regenerated = guarded_answer(context, client=object())

    assert (verdict.passed, regenerated) == (True, False)
    assert text == "Chicken Bake is about 3,470 kcal."


def test_guarded_answer_keeps_a_successful_retry(monkeypatch):
    """The regeneration must be given the complaint and its result used when it passes."""
    context = context_of()
    seen = {}

    def fake(ctx, client=None, note="", on_text=None):
        seen["note"] = note
        return "It is 9,876 kcal." if not note else "Chicken Bake is about 3,470 kcal."

    monkeypatch.setattr("pantryiq.agent.generate.answer", fake)

    text, verdict, regenerated = guarded_answer(context, client=object())

    assert (verdict.passed, regenerated) == (True, True)
    assert "9,876" in seen["note"], "the retry was not told what was wrong"
    assert text == "Chicken Bake is about 3,470 kcal."


def test_a_bare_decimal_is_extracted():
    """`.5` — removing the `|\\.\\d+` branch from _NUMBER survived the sweep, and a fabricated
    number the extractor cannot see is a number the guardrail cannot check."""
    assert extract_numbers("about .5 of a cup") == [0.5]
    assert not check("It is .5 kcal per gram.", context_of()).passed


def test_a_field_name_shown_in_the_prompt_is_not_a_numeric_claim():
    """`identifiers` existed for exactly this and was wired up only in explain.py, so a user
    answer quoting `kcal_per_100g` back had its 100 read as a fabrication."""
    from pantryiq.agent.context import PROMPT_FIELD_NAMES

    context = AnswerContext(question="q", query=PantryQuery(),
                            recipes=(RecipeFact.from_candidate(make_candidate()),),
                            identifiers=PROMPT_FIELD_NAMES)

    assert check("Chicken Bake reports kcal_per_100g of 172.", context).passed
