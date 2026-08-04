"""Structured retrieval — the matching rule, the quality bar, and the shape of the SQL.

The word-boundary rule was chosen by measurement (see the module docstring); these pin the
behaviours that measurement bought, so a later "simplification" back to substring or exact
matching fails loudly rather than quietly changing what the agent is allowed to say.
"""
import time
from pathlib import Path

import duckdb
import pytest

from pantryiq.agent.retrieval import PantryQuery, retrieve, term_pattern

# (recipe_id, title, lines, coverage, trust, kcal, cost, tags)
CORPUS = [
    ("r1", "Egg Salad",      ["egg", "mayonnaise", "celery"],  1.0,  0.90, 400.0, 3.0, ["vegetarian"]),
    ("r2", "Eggplant Bake",  ["eggplant", "tomato", "cheese"], 1.0,  0.80, 600.0, 5.0, ["vegetarian"]),
    ("r3", "Scrambled Eggs", ["eggs yolk", "butter"],          1.0,  0.70, 300.0, 2.0, []),
    ("r4", "Thin Gruel",     ["egg", "water"],                 0.50, 0.95, 100.0, 1.0, []),
    ("r5", "Egg Curry",      ["egg", "onion", "tomato"],       0.90, 0.60, 800.0, 9.0,
     ["vegetarian", "gluten-free"]),
]


@pytest.fixture
def db(tmp_path):
    """A miniature Gold with the four tables retrieval reads.

    Not named `gold.duckdb`: DuckDB names the catalog after the file, and a `gold` catalog
    holding a `gold` schema makes every `gold.<table>` reference ambiguous.
    """
    path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA gold")
    con.execute("CREATE TABLE gold.recipe_ingredients_resolved (recipe_id VARCHAR, "
                "normalized_text VARCHAR)")
    con.execute("CREATE TABLE gold.recipe_meta (recipe_id VARCHAR, title VARCHAR)")
    con.execute("CREATE TABLE gold.recipe_tags (recipe_id VARCHAR, tag VARCHAR)")
    con.execute("""
        CREATE TABLE gold.recipe_nutrition (
            recipe_id VARCHAR, ingredient_count INTEGER, counted_ingredients INTEGER,
            total_kcal DOUBLE, kcal_per_100g DOUBLE, total_grams DOUBLE,
            total_protein_g DOUBLE, total_fat_g DOUBLE, total_carb_g DOUBLE,
            servings INTEGER, kcal_per_serving DOUBLE,
            cost_total_usd DOUBLE, cost_coverage DOUBLE,
            nutrition_coverage DOUBLE, data_trust_score DOUBLE)
    """)
    for recipe_id, title, lines, coverage, trust, kcal, cost, tags in CORPUS:
        con.execute("INSERT INTO gold.recipe_meta VALUES (?, ?)", [recipe_id, title])
        for text in lines:
            con.execute("INSERT INTO gold.recipe_ingredients_resolved VALUES (?, ?)",
                        [recipe_id, text])
        for tag in tags:
            con.execute("INSERT INTO gold.recipe_tags VALUES (?, ?)", [recipe_id, tag])
        con.execute("INSERT INTO gold.recipe_nutrition VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "NULL, NULL, ?, 1.0, ?, ?)",
                    [recipe_id, len(lines), len(lines), kcal, 100.0, 400.0,
                     10.0, 5.0, 20.0, cost, coverage, trust])
    con.close()
    return path


def ids(candidates):
    return [candidate.recipe_id for candidate in candidates]


def test_a_pantry_term_does_not_match_a_longer_word_containing_it(db):
    """The rule this module exists to get right. Substring matching sends `egg` into `eggplant`
    (and `corn` into `popcorn`, `acorn`, `mexicorn` on the real corpus), which puts a food the
    user did not ask for into an answer that claims it did."""
    found = ids(retrieve(PantryQuery(pantry=("egg",), min_coverage=0.0), db))

    assert "r2" not in found, "eggplant matched a query for egg"


def test_a_pantry_term_matches_its_own_plural(db):
    """The other half of the same rule: exact matching finds `chicken` in 211 recipes against
    1,565 that mention it, because the corpus writes `eggs`, `onions`, `carrots`."""
    found = ids(retrieve(PantryQuery(pantry=("egg",), min_coverage=0.0), db))

    assert "r3" in found, "'eggs yolk' did not match a query for egg"


def test_regex_metacharacters_in_a_pantry_term_are_stripped(db):
    """Pantry terms are user text interpolated into a regex. Unsanitised, `.*` matches every line
    in the corpus and the user silently gets a ranking over everything."""
    assert term_pattern("chicken|.*") == r"\bchickens?\b"


def test_a_pantry_term_that_sanitises_to_nothing_returns_nothing(db):
    """Caught by the test above: `.*` strips to an empty pattern, which then looked identical to
    an empty pantry and fell through to the whole-corpus trust ranking. Naming no ingredients and
    naming only ingredients we could not read are different requests, and answering the second
    with everything reads as comprehension the agent does not have."""
    assert retrieve(PantryQuery(pantry=(".*",), min_coverage=0.0), db) == []


def test_recipes_below_the_coverage_bar_are_excluded(db):
    """A recipe whose nutrition is half guessed cannot honestly answer a nutrition question.
    r4 has the highest trust score in the fixture and must still be dropped at 0.5 coverage."""
    found = ids(retrieve(PantryQuery(pantry=("egg",), min_coverage=0.8), db))

    assert "r4" not in found


def test_every_requested_tag_must_be_present_not_merely_one(db):
    """'vegetarian and gluten-free' means both. Matching any one tag would return recipes that
    satisfy neither constraint the user actually stated."""
    both = ids(retrieve(PantryQuery(pantry=("egg",), tags=("vegetarian", "gluten-free"),
                                    min_coverage=0.0), db))

    assert both == ["r5"], "a recipe carrying only one of the two requested tags came back"


def test_an_excluded_term_removes_the_whole_recipe(db):
    """Exclusion is a property of the recipe, not the line: a recipe containing onion anywhere
    is unusable to someone avoiding onion, however well the rest of it matches."""
    found = ids(retrieve(PantryQuery(pantry=("egg", "tomato"), exclude=("onion",),
                                     min_coverage=0.0), db))

    assert "r5" not in found


def test_ranking_is_by_count_of_matched_terms_then_trust(db):
    """The ordering has to be explainable — 'it uses more of what you have' — because the answer
    quotes the reason. r5 matches two terms at trust 0.60; r1 matches one at 0.90."""
    found = retrieve(PantryQuery(pantry=("egg", "tomato"), min_coverage=0.0), db)

    assert found[0].recipe_id == "r5"
    assert found[0].matched == ("egg", "tomato")
    assert ids(found).index("r1") < ids(found).index("r3"), "trust did not break the 1-match tie"


def test_matched_terms_come_back_as_the_user_typed_them(db):
    """The answer says 'has egg', not r'has \\begg s?\\b'. The regex is an implementation detail
    and must not reach a generated response."""
    found = retrieve(PantryQuery(pantry=("egg", "celery"), min_coverage=0.0), db)

    assert found[0].matched == ("celery", "egg")
    assert found[0].missing == ()


def test_terms_the_recipe_lacks_are_reported_as_missing(db):
    """Retrieval returns partial matches on purpose — a recipe using three of your four things
    is a useful answer as long as the answer says which one you are short of."""
    found = retrieve(PantryQuery(pantry=("egg", "saffron"), min_coverage=0.0), db)

    assert found[0].missing == ("saffron",)


def test_a_query_with_no_pantry_terms_still_returns_candidates(db):
    """'Something vegetarian under 500 kcal' names no ingredient. Retrieval returning nothing
    would push the agent to answer from outside the warehouse, which is the one thing it must
    never do — so an unfiltered query ranks the whole corpus by trust instead."""
    found = retrieve(PantryQuery(pantry=(), max_kcal=500, min_coverage=0.0), db)

    assert ids(found) == ["r4", "r1", "r3"], "no-pantry query did not fall back to a trust ranking"
    assert all(candidate.matched == () for candidate in found)


def test_a_kcal_bound_applies_to_the_basis_the_user_meant(db):
    """'Under 500 calories' means a serving, not a whole casserole. On the real corpus the median
    recipe is 2,539 kcal total, so applying a 500 bound to the total returns dips and dressings
    while applying it per serving returns meals. The fixture mirrors that: every recipe passes a
    500-kcal total bound, and none has a per-serving figure at all."""
    by_total = retrieve(PantryQuery(pantry=("egg",), max_kcal=500, min_coverage=0.0), db)
    by_serving = retrieve(PantryQuery(pantry=("egg",), max_kcal=500, kcal_basis="serving",
                                      min_coverage=0.0), db)

    assert by_total, "a total-basis bound should match the fixture's small recipes"
    assert by_serving == [], "a per-serving bound must not match rows with no per-serving figure"


def test_an_unknown_kcal_basis_is_refused_rather_than_interpolated(db):
    """`kcal_basis` picks a column name, so it is the one field in the query that reaches the SQL
    as text rather than a bound parameter. It is checked against a closed set."""
    with pytest.raises(ValueError, match="kcal_basis"):
        retrieve(PantryQuery(pantry=("egg",), kcal_basis="total; DROP TABLE gold.recipe_meta"), db)


@pytest.mark.skipif(not Path("data/pantryiq_gold.duckdb").exists(), reason="export not present")
def test_patterns_are_bound_as_constants_rather_than_joined_as_a_column():
    """A performance regression that changes what the agent can do inside its latency budget.

    Joining `gold.recipe_ingredients_resolved` against a table of patterns reads better and makes
    the pattern a *column*, so DuckDB recompiles the regex for each of the 112,463 rows: three
    terms cost 973 ms that way against 17 ms as bound constants, and one regex alone costs 4.0 ms.
    The ceiling here is loose enough not to flake on a busy machine and far below the join form.
    """
    query = PantryQuery(pantry=("chicken", "rice", "onion"))
    retrieve(query)  # warm the page cache; the first connect also pays ~4 ms

    started = time.perf_counter()
    retrieve(query)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert elapsed_ms < 250, f"retrieval took {elapsed_ms:.0f} ms — is the pattern a column again?"
