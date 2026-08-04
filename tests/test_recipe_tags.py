"""Dietary tags — the one thing this project publishes that could hurt someone if it is wrong.

**Most of these now run everywhere**, against a synthetic warehouse that `tests/fixtures/
warehouse.py` builds and that **dbt builds the real Gold models on top of**. That last part is
what keeps them honest: the tags under test come from `models/gold/recipe_tags.sql` exactly as in
production, from ten recipes whose right answer is known by construction. Asserting against tags
the fixture wrote would be a tautology.

Until this existed the whole file was gated on a local export, so nine safety assertions ran on
the author's laptop and nowhere else — and three mutations to the tag SQL left the entire suite
green, because `make test` never invokes dbt.

The tests below the divider still need the real corpus, and say why. They are regression pins on
specific recipes and measured leakage rates, which a synthetic warehouse cannot stand in for.
"""
import json
import re
from pathlib import Path

import duckdb
import pytest

EXPORT = Path("data/pantryiq_gold.duckdb")
needs_corpus = pytest.mark.skipif(not EXPORT.exists(), reason="real corpus not present")

# Deliberately NOT the pattern the tag rule uses. Titles and directions are independent of the
# ingredient list the rule reads, which is what makes them a check rather than a restatement.
MEAT_IN_TITLE = (r"\b(beef|pork|chicken|turkey|ham|bacon|sausage|meat ?loaf|meatball|steak|lamb"
                 r"|veal|burger|hamburger|brisket|salami|pepperoni|venison|bologna|jerky)\b")


def tags_for(con, recipe_id: str) -> set[str]:
    return {row[0] for row in con.execute(
        "SELECT tag FROM gold.recipe_tags WHERE recipe_id = ?", [recipe_id]).fetchall()}


# ----------------------------------------------------------- the rule, on known answers

def test_a_recipe_of_qualifying_ingredients_gets_every_tag(fixture_gold):
    """The control. Without it, a rule that tags nothing would pass every test below."""
    assert tags_for(fixture_gold, "r_vegan") == {"vegetarian", "vegan", "gluten-free"}


def test_dairy_disqualifies_vegan_but_not_vegetarian(fixture_gold):
    assert tags_for(fixture_gold, "r_dairy") == {"vegetarian", "gluten-free"}


def test_wheat_disqualifies_gluten_free_but_not_vegetarian(fixture_gold):
    assert tags_for(fixture_gold, "r_wheat") == {"vegetarian", "vegan"}


def test_an_ingredient_classified_no_disqualifies_the_tag(fixture_gold):
    """Worcestershire sauce contains anchovies. It has no meat word in its name and sits in an
    allowed food group, so it defeated both previous rules — a name denylist and then a
    food-group allowlist. It is the reason the question is now asked per entity."""
    assert tags_for(fixture_gold, "r_worcester") == set()


def test_an_unknown_classification_disqualifies_exactly_like_a_no(fixture_gold):
    """705 of 8,187 entities are genuinely unclassifiable from their description. Silence is the
    safe answer — and SQL makes this easy to get wrong, because `bool_and` SKIPS nulls."""
    assert tags_for(fixture_gold, "r_unknown") == set()


def test_a_recipe_below_full_coverage_gets_no_tag(fixture_gold):
    """The ingredient nobody could identify is exactly the one that might disqualify it."""
    assert tags_for(fixture_gold, "r_partial") == set()


def test_the_directions_veto_fires_when_the_method_names_a_missing_food(fixture_gold):
    """`coverage = 1.0` means "every line we have was weighed", not "we have every line". This
    recipe's ingredients are onion and vinegar; its method says "remove skin from 2 rings
    sausage". Gluten-free survives, because sausage is not gluten."""
    assert tags_for(fixture_gold, "r_truncated") == {"gluten-free"}


def test_the_recipes_own_words_veto_gluten_even_when_the_entity_qualifies(fixture_gold):
    """The subtlest branch. The line reads `flour` and resolved to *Millet flour*, which is
    genuinely gluten-free — so the classifier was right and the tag would still have been false.
    Vegetarian and vegan survive; gluten-free does not."""
    assert tags_for(fixture_gold, "r_millet") == {"vegetarian", "vegan"}


def test_the_name_denylist_catches_entity_resolution_being_wrong(fixture_gold):
    """The classifier judges the ENTITY, and entity resolution is 67.3% accurate. A line reading
    `chicken broth` that resolved to cider vinegar is vegetarian by every entity-level test."""
    assert tags_for(fixture_gold, "r_nameflag") == {"gluten-free"}


def test_no_tagged_recipe_contains_a_non_qualifying_entity(fixture_gold):
    """The property stated over the whole fixture rather than recipe by recipe."""
    for tag, column in (("vegetarian", "is_vegetarian"), ("vegan", "is_vegan"),
                        ("gluten-free", "is_gluten_free")):
        leaked = fixture_gold.execute(f"""
            SELECT count(*) FROM gold.recipe_tags t
            JOIN gold.recipe_ingredients_resolved r USING (recipe_id)
            JOIN gold.canonical_ingredients ci ON ci.canonical_id = r.canonical_id
            WHERE t.tag = ? AND coalesce(ci.{column}, 'unknown') <> 'yes'
        """, [tag]).fetchone()[0]

        assert leaked == 0, f"{tag} admitted a non-qualifying entity"


# --------------------------------------------------- regression pins on the real corpus

@pytest.fixture(scope="module")
def con():
    if not EXPORT.exists():
        pytest.skip("real corpus not present")
    connection = duckdb.connect(str(EXPORT), read_only=True)
    yield connection
    connection.close()


@needs_corpus
def test_the_two_recipes_found_by_inspection_are_no_longer_tagged(con):
    """Both were fully covered, fully resolved, and wrong — found by joining Phase 4's recipe
    titles against the tags, a signal that did not exist when the old predicate was written."""
    for recipe_id in ("recipenlg:9231", "recipenlg:10493"):
        tags = tags_for(con, recipe_id)

        assert "vegetarian" not in tags and "vegan" not in tags, f"{recipe_id} is a meat loaf"


@needs_corpus
def test_residual_title_leakage_stays_within_what_was_measured(con):
    """Measured with titles, which the rule does not read — the vetoes use directions and
    ingredient text instead, precisely so this stays independent of the fix.

    Three vegetarian titles still name meat and all three are artifacts of this test rather than
    the tags: "Cucumber Sauce **For** Fish", "Cold Pack Fish" (vinegar and salt) and "Fish Fry
    Coating Mix" (cornmeal and spices) contain no fish. The bound is what is asserted.
    """
    leaked = con.execute("""
        SELECT count(DISTINCT t.recipe_id) FROM gold.recipe_tags t
        JOIN gold.recipe_meta m USING (recipe_id)
        WHERE t.tag = 'vegetarian' AND regexp_matches(lower(m.title), ?)
    """, [MEAT_IN_TITLE]).fetchone()[0]

    assert leaked <= 3, f"title leakage rose to {leaked} (was 13 under the group allowlist, 3 now)"


@needs_corpus
def test_no_gluten_free_recipe_names_unqualified_grain_in_its_own_lines(con):
    leaked = con.execute("""
        SELECT count(DISTINCT t.recipe_id) FROM gold.recipe_tags t
        JOIN gold.recipe_ingredients_resolved r USING (recipe_id)
        WHERE t.tag = 'gluten-free'
          AND regexp_matches(lower(r.normalized_text),
              '\\b(flour|bread|oats|wheat|barley|rye|pasta|noodle|macaroni|graham|biscuit)\\w*\\b')
          AND NOT regexp_matches(lower(r.normalized_text),
              '\\b(rice|corn|almond|coconut|millet|chestnut|arrowroot|tapioca|buckwheat|potato)\\w*\\b')
    """).fetchone()[0]

    assert leaked == 0


def test_the_directions_signal_can_actually_fire():
    """The control, and it is not hypothetical: the first version of this measurement matched 0
    of 15,000 recipes because `directions` is stored as a JSON *string* and was being joined as
    if it were a list, spacing out every character. A check that cannot fire proves nothing."""
    silver = Path("data/pantryiq.duckdb")
    if not silver.exists():
        pytest.skip("real corpus not present")
    connection = duckdb.connect(str(silver), read_only=True)
    try:
        fires, total = connection.execute(
            "SELECT count(*) FILTER (WHERE regexp_matches(coalesce(directions, ''), ?)), count(*)"
            " FROM silver.recipe_meta", [MEAT_IN_TITLE]).fetchone()
    finally:
        connection.close()

    assert fires > 0.1 * total, f"the directions pattern only fires on {fires}/{total} recipes"


def test_bronze_really_does_ship_truncated_ingredient_lists():
    """The finding behind the veto, pinned so it is not re-litigated: this is a source-data gap,
    not a parser bug. Silver kept every line Bronze had."""
    lakehouse = Path("data/lakehouse")
    if not lakehouse.exists():
        pytest.skip("lakehouse not present")
    from pantryiq.lakehouse.catalog import get_catalog, scan_with_duckdb

    table = scan_with_duckdb(get_catalog().load_table("bronze.raw_recipes"))
    ids = table.column("recipe_id").to_pylist()
    payload = json.loads(table.column("raw_payload").to_pylist()[ids.index("recipenlg:570")])

    # Both `ingredients` and `directions` are stored as JSON *strings*, not lists — the same trap
    # that made the first directions measurement match 0 of 15,000 recipes.
    lines = json.loads(payload["ingredients"])

    assert not re.search(r"bologna", " ".join(lines), re.IGNORECASE)
    assert len(lines) == 4
