"""Tests that close the mutation-sweep survivors in guardrail.py and context.py.

Each one was validated twice: it passes against unmutated HEAD, and it fails against the single
mutation named in its docstring. Nothing here is satisfied by an incidental fixture property.
"""
from pantryiq.agent.context import AnswerContext, RecipeFact
from pantryiq.agent.guardrail import check, permitted
from pantryiq.agent.retrieval import PantryQuery
from test_context import make_candidate
from test_guardrail import context_of, two_recipes


def ctx_with(identifiers=(), **overrides) -> AnswerContext:
    return AnswerContext(question="q", query=PantryQuery(pantry=("chicken", "rice")),
                         recipes=(RecipeFact.from_candidate(make_candidate(**overrides)),),
                         identifiers=identifiers)


# --------------------------------------------------------------- sentence splitting

def test_a_newline_ends_a_sentence_so_a_bulleted_answer_is_scoped_per_line():
    """Kills: dropping `|\\n+` from `_SENTENCE`. A bulleted answer has no terminal punctuation,
    so without the newline branch every line merges into one sentence, every recipe lands in
    scope at once, and per-recipe scoping silently degrades to the old union."""
    text = "Favorite Chicken uses chicken and rice\nBroccoli Casserole is about 3,470 kcal."

    assert not check(text, two_recipes()).passed


# --------------------------------------------------------------- identifier stripping

def test_a_purely_numeric_identifier_is_never_stripped():
    """Kills: dropping the `not` from `_strippable`, and dropping the guard altogether. A
    numeric identifier reaching this channel from explain.py would delete a real quantity from
    the response, making "9999 rows were dropped" invisible to the guardrail."""
    assert not check("9999 rows were dropped.", ctx_with(identifiers=("9999",))).passed


def test_identifiers_are_stripped_before_ranges_are_split():
    """Kills: reordering `_scan` to range-split first. Splitting `60-69` into `60 69` first
    destroys the literal the `str.replace` is looking for, leaving both digits in the response
    as unexplained numbers — the exact case the docstring cites."""
    context = context_of(make_candidate(title="Chocolate, dark, 60-69% cacao"))

    assert check("Chocolate, dark, 60-69% cacao is about 3,470 kcal.", context).passed


# --------------------------------------------------------------- list markers

def test_a_list_marker_is_only_recognised_at_the_start_of_a_line():
    """Kills: dropping the `^` anchor from `_LIST_MARKER`. `positional, not vocabulary` is only
    true while the position is anchored; unanchored, any sentence ending on a digit donates a
    free pass to the number before the full stop."""
    assert not check("The whole dish costs $3. That is cheap.", context_of()).passed


def test_a_multi_digit_integer_is_not_a_list_marker():
    """Kills: widening `\\d{1,2}` to `\\d{1,4}`. The two-digit bound is what stops a fabricated
    figure from laundering itself through the marker bypass."""
    assert not check("9999. Chicken Bake is a good option.", context_of()).passed


# --------------------------------------------------------------- permitted()

def test_a_count_gets_no_tolerance_where_one_percent_would_exceed_one():
    """Kills: applying the band to `exact` as well as `measured`. The existing
    `test_a_count_gets_no_tolerance_at_any_magnitude` claims "at any magnitude" but asserts only
    at 5 and 6, where 1% of 5 is 0.05 and cannot bridge the gap — so it stays green with the
    band applied. Counts only need protecting once 1% of them exceeds 1."""
    assert not permitted(101.0, exact={100.0}, measured=set()), "101 of 100 ingredients weighed"


def test_the_tolerance_boundary_is_inclusive():
    """Kills: `<=` -> `<` in the band comparison."""
    assert permitted(101.0, exact=set(), measured={100.0})


def test_the_band_is_scaled_by_the_stored_value_not_the_quoted_one():
    """Kills: `max(abs(candidate), 1.0)` -> `max(abs(value), 1.0)`. Scaling by the number the
    model wrote lets the model widen its own tolerance by quoting high."""
    assert not check("roughly 3,505 kcal", context_of()).passed


# --------------------------------------------------------------- title matching

def test_a_parenthetical_title_is_matched_when_another_recipe_was_named_first():
    """Kills: `split("(")[0]` -> `[-1]`, and dropping `trimmed` from `_needles`.

    The existing `test_a_title_with_a_nested_parenthetical_still_matches` cannot fail: when the
    title stops matching, `mentions` returns empty and `check` falls back to the whole-context
    union, which still contains the value. Naming a different recipe first removes that rescue —
    and it is the failure the docstring actually describes ("attributed to the recipe named
    before it"). 5,555 rather than 2,004 because 2,004 is inside 1% of the fixture's default
    total_grams of 2,017, a second incidental rescue."""
    context = context_of(
        make_candidate(recipe_id="r1", title="Baked Oatmeal", total_kcal=1364),
        make_candidate(recipe_id="r2", total_kcal=5555,
                       title="Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies)"))
    text = "Baked Oatmeal is about 1,364 kcal. Honey Oatmeal Drop Cookies is about 5,555 kcal."

    assert check(text, context).passed


def test_a_recipe_named_only_by_its_id_is_scoped_to_that_recipe():
    """Kills: dropping `recipe.recipe_id` from `_needles`, and raising the length floor to 4.
    No existing test can see either, because every fixture id is `r1`/`r2` — two characters,
    already below the `len(needle) >= 3` floor, so id matching is inert throughout the suite."""
    context = context_of(
        make_candidate(recipe_id="recipenlg:14797", title="Baked Oatmeal", total_kcal=1364),
        make_candidate(recipe_id="rid", title="Broccoli Casserole", total_kcal=2004))

    assert not check("Baked Oatmeal is good. rid is about 1,364 kcal.", context).passed


def test_a_title_is_matched_case_insensitively():
    """Kills: dropping `re.IGNORECASE` from `mentions`. A single-recipe context cannot show
    this — losing the match falls back to the union, which still holds the value."""
    assert not check("broccoli casserole is about 3,470 kcal.", two_recipes()).passed


# --------------------------------------------------------------- overlap resolution

def test_the_longest_title_wins_when_one_title_is_a_prefix_of_another():
    """Kills: reversing the sort key to prefer the shortest name, and dropping the sort.

    The existing `test_a_title_that_contains_another_title_does_not_split_it` uses titles that
    overlap in the MIDDLE, so the two candidates have different start offsets and the tiebreak
    never runs. The tiebreak only fires when one title is a prefix of the other."""
    context = context_of(
        make_candidate(recipe_id="r1", title="Potato Casserole", total_kcal=6982),
        make_candidate(recipe_id="r2", title="Potato Casserole Deluxe", total_kcal=1234))

    assert check("Potato Casserole Deluxe is about 1,234 kcal.", context).passed


def test_a_title_inside_another_title_does_not_widen_the_scope():
    """Kills: `covered_to = end` -> `start`.

    The negative direction the existing overlap test omits. A spurious extra mention can only
    ever WIDEN a sentence's scope, so a passing assertion cannot detect one; it takes a number
    that belongs to the inner recipe and must be refused for the outer one."""
    context = context_of(
        make_candidate(recipe_id="r1", title="Hash Brown Potato Casserole", total_kcal=4459),
        make_candidate(recipe_id="r2", title="Potato Casserole", total_kcal=6982))

    assert not check("Hash Brown Potato Casserole is about 6,982 kcal.", context).passed


def test_two_titles_that_abut_are_both_in_scope():
    """Kills: `start < covered_to` -> `<=`, which drops a mention beginning exactly where the
    previous one ended."""
    context = context_of(
        make_candidate(recipe_id="r1", title="Favorite Chicken", total_kcal=3470),
        make_candidate(recipe_id="r2", title="Broccoli Casserole", total_kcal=1968))

    assert check("Favorite ChickenBroccoli Casserole: 1,968 kcal.", context).passed


# --------------------------------------------------------------- the scoping loop

def test_a_sentence_naming_no_recipe_inherits_the_last_one_named():
    """Kills: dropping the `current` carry-forward. Losing it does not fail open loudly — the
    sentence falls through to the answer-level branch, whose ledger is every recipe's numbers
    unioned together, so the carry-forward's absence looks exactly like the union bug this
    module was rebuilt to remove. Only a misattribution can tell the two apart."""
    text = "Favorite Chicken is your best match. It is about 1,968 kcal."

    assert not check(text, two_recipes()).passed


def test_the_carried_recipe_is_the_last_one_named_in_the_sentence():
    """Kills: `current = here[-1]` -> `here[0]`. After a sentence naming two dishes, the
    follow-on line belongs to the second."""
    text = ("Favorite Chicken and Broccoli Casserole are both options. "
            "The latter is about 1,968 kcal.")

    assert check(text, two_recipes()).passed


def test_a_number_fabricated_twice_is_reported_once():
    """Kills: dropping `dict.fromkeys` from the `unsupported` tuple. `complaint()` reads this
    straight into the regeneration prompt, and a repeated number lists twice."""
    verdict = check("It is 9,999 kcal, and 9,999 kcal is a lot.", context_of())

    assert verdict.unsupported == (9999.0,)


# --------------------------------------------------------------- context.py

def test_costed_ingredients_is_coverage_times_the_full_ingredient_count():
    """Kills: deriving it from `counted_ingredients`, and dropping the derivation entirely.

    The derivation is invisible to the whole suite today. The fixture has cost_coverage 0.4 and
    ingredient_count 5, so the derived value is 2.0 — which is also `len(matched)`, already in
    `exact_numbers()`. Removing the derivation changes nothing the fixture can see, and
    `round(0.4 * 4)` is 2.0 as well, so the wrong denominator is invisible too."""
    fact = RecipeFact.from_candidate(make_candidate(
        cost_coverage=0.75, ingredient_count=8, counted_ingredients=4,
        matched=("chicken",), missing=(), servings=None))

    assert 6.0 in fact.exact_numbers(), "0.75 of 8 ingredients priced"
    assert 3.0 not in fact.exact_numbers(), "not 0.75 of the WEIGHED count"
