"""Dietary tags — the one thing this project publishes that could hurt someone if it is wrong.

These run against the published export rather than a fixture. The tag rule is dbt SQL over the
real entity set, and the failures it exists to prevent were all cases where the *data* defeated
a plausible-looking predicate — which a hand-built fixture would not have reproduced.
"""
import json
import re
from pathlib import Path

import duckdb
import pytest

EXPORT = Path("data/pantryiq_gold.duckdb")
pytestmark = pytest.mark.skipif(not EXPORT.exists(), reason="export not present")

# Deliberately NOT the pattern the tag rule uses. Titles and directions are independent of the
# ingredient list the rule reads, which is what makes them a check rather than a restatement.
MEAT_IN_TITLE = (r"\b(beef|pork|chicken|turkey|ham|bacon|sausage|meat ?loaf|meatball|steak|lamb"
                 r"|veal|burger|hamburger|brisket|salami|pepperoni|venison|bologna|jerky)\b")


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect(str(EXPORT), read_only=True)
    yield connection
    connection.close()


def test_a_tag_requires_every_ingredient_to_sit_in_an_allowed_food_group(con):
    """The allowlist, stated as a property. No vegetarian recipe may contain an ingredient from
    a flesh food group — this is what a denylist over product names could not guarantee, because
    "BURGER KING, Hamburger" contains no meat word at all."""
    leaked = con.execute("""
        SELECT DISTINCT m.title, ci.display_name, ci.food_category
        FROM gold.recipe_tags t
        JOIN gold.recipe_ingredients_resolved r USING (recipe_id)
        JOIN gold.canonical_ingredients ci ON ci.canonical_id = r.canonical_id
        LEFT JOIN gold.recipe_meta m USING (recipe_id)
        WHERE t.tag = 'vegetarian' AND ci.food_category IN (
            'Beef Products', 'Pork Products', 'Poultry Products',
            'Lamb, Veal, and Game Products', 'Finfish and Shellfish Products',
            'Sausages and Luncheon Meats', 'Fast Foods', 'Restaurant Foods')
    """).fetchall()

    assert leaked == [], f"vegetarian recipes containing a flesh food group: {leaked[:5]}"


def test_the_two_recipes_found_by_inspection_are_no_longer_tagged(con):
    """Regression pins. Both were fully covered, fully resolved, and wrong — found by joining
    Phase 4's new recipe titles against the tags, a signal that did not exist when the old
    predicate was written."""
    for recipe_id in ("recipenlg:9231", "recipenlg:10493"):
        tags = con.execute("SELECT tag FROM gold.recipe_tags WHERE recipe_id = ?",
                           [recipe_id]).fetchall()

        assert tags == [], f"{recipe_id} is tagged {tags} — it is a meat loaf"


def test_an_unknown_food_group_disqualifies_rather_than_being_ignored(con):
    """1.0% of foods carry no USDA category. Under an allowlist an unknown must fail, and SQL
    makes that easy to get wrong: `bool_and` skips NULLs, so an uncategorised ingredient would
    silently not count against the recipe."""
    leaked = con.execute("""
        SELECT count(*) FROM gold.recipe_tags t
        JOIN gold.recipe_ingredients_resolved r USING (recipe_id)
        JOIN gold.canonical_ingredients ci ON ci.canonical_id = r.canonical_id
        WHERE t.tag IN ('vegetarian', 'vegan') AND ci.food_category IS NULL
    """).fetchone()[0]

    assert leaked == 0


def test_gluten_free_excludes_the_grain_food_groups(con):
    leaked = con.execute("""
        SELECT DISTINCT ci.food_category FROM gold.recipe_tags t
        JOIN gold.recipe_ingredients_resolved r USING (recipe_id)
        JOIN gold.canonical_ingredients ci ON ci.canonical_id = r.canonical_id
        WHERE t.tag = 'gluten-free' AND ci.food_category IN (
            'Baked Products', 'Cereal Grains and Pasta', 'Breakfast Cereals')
    """).fetchall()

    assert leaked == []


def test_vegan_admits_no_dairy_or_egg(con):
    leaked = con.execute("""
        SELECT count(*) FROM gold.recipe_tags t
        JOIN gold.recipe_ingredients_resolved r USING (recipe_id)
        JOIN gold.canonical_ingredients ci ON ci.canonical_id = r.canonical_id
        WHERE t.tag = 'vegan' AND ci.food_category = 'Dairy and Egg Products'
    """).fetchone()[0]

    assert leaked == 0


def test_residual_title_leakage_stays_within_what_was_measured(con):
    """Measured with titles, which the tag rule does not read — the veto uses directions instead,
    precisely so this check stays independent of the fix.

    5 vegetarian titles still name meat, and at least three are artifacts of this test rather
    than the tags: "Cucumber Sauce For Fish", "Fish Fry Coating Mix" and "Meat Marinade" are
    condiments that contain no meat. The bound is what is asserted, not zero.
    """
    leaked = con.execute("""
        SELECT count(DISTINCT t.recipe_id) FROM gold.recipe_tags t
        JOIN gold.recipe_meta m USING (recipe_id)
        WHERE t.tag = 'vegetarian' AND regexp_matches(lower(m.title), ?)
    """, [MEAT_IN_TITLE]).fetchone()[0]

    assert leaked <= 5, f"title leakage rose to {leaked} (was 13 under the denylist, 5 now)"


def test_a_truncated_ingredient_list_is_vetoed_by_the_directions():
    """`nutrition_coverage = 1.0` means "every line we have was weighed", not "we have every
    line". Bronze's Pickled Bologna lists vinegar, sugar, salt and pickling spice and no bologna,
    so it passes every ingredient-based test and is not vegan. The method names it.
    """
    silver = Path("data/pantryiq.duckdb")
    if not silver.exists():
        pytest.skip("warehouse not present")
    connection = duckdb.connect(str(silver), read_only=True)
    try:
        directions = connection.execute(
            "SELECT directions FROM silver.recipe_meta WHERE recipe_id = 'recipenlg:570'"
        ).fetchone()[0]
    finally:
        connection.close()

    assert "bologna" in directions, "the directions no longer carry the missing ingredient"

    con = duckdb.connect(str(EXPORT), read_only=True)
    try:
        tags = con.execute(
            "SELECT tag FROM gold.recipe_tags WHERE recipe_id = 'recipenlg:570'").fetchall()
    finally:
        con.close()

    assert tags == [], "Pickled Bologna is tagged despite its method naming bologna"


def test_the_directions_signal_can_actually_fire():
    """The control, and it is not hypothetical: the first version of this measurement matched 0
    of 15,000 recipes because `directions` is stored as a JSON *string* and was being joined as
    if it were a list, spacing out every character. A check that cannot fire proves nothing."""
    silver = Path("data/pantryiq.duckdb")
    if not silver.exists():
        pytest.skip("warehouse not present")
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
