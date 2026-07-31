"""Parser/normalizer tests for Silver ingredient lines — the tricky real RecipeNLG shapes."""
import pytest

from pantryiq.silver.ingredient_lines import (
    build_line_rows,
    normalize,
    parse_line,
    singularize,
    stratum,
    write_silver,
)

# (raw line, quantity, unit, normalized_text) — every case is a real line shape from Bronze.
CASES = [
    ("1 c. firmly packed brown sugar", 1.0, "cup", "brown sugar"),
    ("1/2 c. evaporated milk", 0.5, "cup", "evaporated milk"),
    ("3 1/2 c. bite size shredded rice biscuits", 3.5, "cup", "bite size rice biscuit"),
    ("2 (16 oz.) pkg. frozen corn", 2.0, "package", "frozen corn"),
    ("1 (8 oz.) can tomato sauce", 1.0, "can", "tomato sauce"),
    ("1/3 c. butter, cubed", 1 / 3, "cup", "butter"),
    ("2 tbsp. butter or margarine", 2.0, "tablespoon", "butter"),
    ("1 large jar Cheez Whiz", 1.0, "jar", "cheez whiz"),
    ("1 large onion", 1.0, None, "onion"),
    ("dash of pepper", None, "dash", "pepper"),
    ("a pinch of salt", 1.0, "pinch", "salt"),
    ("juice of 1 lemon", None, None, "juice of lemon"),
    ("1 1/2 c. semi-sweet chocolate mini morsels, divided", 1.5, "cup", "semi-sweet chocolate mini morsel"),  # noqa: E501
    ("2 or 3 hot jalapeno or Serrano chilies, stemmed, seeded and finely minced", 2.0, None, "hot jalapeno"),  # noqa: E501
    ("1 pkg. dry yeast (5 tsp.)", 1.0, "package", "dry yeast"),
    ("1/2 head cabbage", 0.5, "head", "cabbage"),
    ("2 beaten eggs", 2.0, None, "egg"),
    ("10 c. water", 10.0, "cup", "water"),
    ("1 can cream of chicken soup", 1.0, "can", "cream of chicken soup"),
    ("3 cloves garlic, minced", 3.0, "clove", "garlic"),
]


@pytest.mark.parametrize("line,quantity,unit,normalized", CASES)
def test_parse_and_normalize(line, quantity, unit, normalized):
    parsed = parse_line(line)
    if quantity is None:
        assert parsed.quantity is None
    else:
        assert parsed.quantity == pytest.approx(quantity)
    assert parsed.unit == unit
    assert normalize(parsed.ingredient_text) == normalized


def test_leading_prep_clause_does_not_erase_the_ingredient():
    """'drained, crushed pineapple' must not normalize to '' just because 'drained' is filler."""
    assert normalize(parse_line("1 c. drained, crushed pineapple").ingredient_text) == "pineapple"
    assert normalize(parse_line("4 c. chopped, cooked chicken").ingredient_text) == "cooked chicken"


def test_chained_second_unit_is_stripped():
    """'6 oz. can lemon juice' must not become a separate entity from 'lemon juice'."""
    assert normalize(parse_line("6 oz. can lemon juice").ingredient_text) == "lemon juice"
    assert normalize(parse_line("1 (8 oz.) pkg. cream cheese").ingredient_text) == "cream cheese"
    # ...but "pound cake" is a food, not a measure.
    assert normalize("pound cake mix") == "pound cake mix"


def test_or_cut_does_not_erase_the_ingredient():
    """The ' or ' cut has the same swallow-the-ingredient failure mode as the comma cut."""
    assert normalize("crushed or chunk pineapple, drained") == "chunk pineapple"
    assert normalize("or more Cheddar cheese, shredded") == "cheddar cheese"


def test_bare_unit_word_is_the_ingredient():
    """A line reading only 'cloves' names the food; consuming it as a unit loses the row."""
    parsed = parse_line("cloves")
    assert parsed.unit is None
    assert normalize(parsed.ingredient_text) == "clove"


def test_percent_is_not_a_quantity():
    """The Phase-1 QTY regex leaked '2% milk'; the parser must not read % as a quantity."""
    assert parse_line("2% milk").quantity is None
    assert parse_line("3.5% milk").quantity is None
    assert normalize(parse_line("2% milk").ingredient_text) == "2% milk"


def test_normalize_keeps_nutritionally_significant_words():
    """Stripping these would change which USDA entity is correct."""
    for text in ("ground beef", "whole milk", "dried oregano", "unsalted butter",
                 "sweetened condensed milk", "frozen peas"):
        kept = normalize(text)
        assert kept.split()[0] == text.split()[0], f"{text} -> {kept}"


def test_singularize_handles_common_shapes():
    assert singularize("tomatoes") == "tomato"
    assert singularize("berries") == "berry"
    assert singularize("eggs") == "egg"
    assert singularize("molasses") == "molasses"  # exception, not "molass"
    assert singularize("asparagus") == "asparagus"
    assert singularize("leaves") == "leaf"  # irregular, not "leave"
    assert singularize("halves") == "half"
    assert singularize("olives") == "olive"  # ...but the generic -s rule still applies


def test_dozen_is_a_unit():
    parsed = parse_line("3 doz. oysters, cleaned and shucked")
    assert (parsed.quantity, parsed.unit) == (3.0, "dozen")
    assert normalize(parsed.ingredient_text) == "oyster"


def test_stratum_thresholds():
    assert stratum(500) == "head"
    assert stratum(20) == "head"
    assert stratum(19) == "mid"
    assert stratum(3) == "mid"
    assert stratum(2) == "tail"
    assert stratum(1) == "tail"


class _FakeBronze:
    """Minimal stand-in for a PyIceberg catalog holding one recipe."""

    def __init__(self, rows):
        self._rows = rows

    def load_table(self, name):
        assert name == "bronze.raw_recipes"
        return self

    def scan(self):
        return self

    def to_arrow(self):
        import pyarrow as pa

        return pa.table(
            {
                "recipe_id": [r[0] for r in self._rows],
                "raw_payload": [r[1] for r in self._rows],
            }
        )


def test_build_and_write_silver(tmp_path):
    payload = '{"ingredients": "[\\"1 c. sugar\\", \\"2 eggs\\", \\"1 c. sugar\\"]"}'
    lines = build_line_rows(_FakeBronze([("recipenlg:0", payload)]))

    assert lines.num_rows == 3
    assert lines.column("normalized_text").to_pylist() == ["sugar", "egg", "sugar"]

    db = write_silver(lines, tmp_path / "pantryiq.duckdb")
    import duckdb

    con = duckdb.connect(str(db))
    distinct = con.execute(
        "SELECT normalized_text, occurrence_count FROM silver.distinct_ingredient_strings "
        "ORDER BY occurrence_count DESC"
    ).fetchall()
    con.close()
    assert distinct == [("sugar", 2), ("egg", 1)]


# Real shapes that were reduced wrongly until 2026-07-31, with what each one cost.
ARTIFACT_CASES = [
    ("salt and pepper to taste", "salt and pepper"),   # "salt and pepper to": 479 occurrences
    ("salt to taste", "salt"),                          # merged into "salt" (4,913)
    ("1 lb. box of powdered sugar", "powdered sugar"),  # leading "of" survived the unit strip
    ("1 (10 oz.) pkg. frozen broccoli cuts", "frozen broccoli"),  # "cut" is FILLER once singular
    ("9-inch pie shell", "inch pie shell"),             # parse_line took the 9, orphaning "-"
    ("1 pkg. Oreo cookies, crushed", "oreo cookie"),    # the -ies rule gave "cooky"
    ("2 c. frozen mixed veggies", "frozen mixed veggie"),
    ("2 cups blueberries", "blueberry"),                # the -ies rule is still right here
    ("1 c. low-fat milk", "low-fat milk"),              # internal hyphens must survive
    ("1 to 2 lb. ground beef", "ground beef"),          # "to" as a range, consumed by _LEAD_QTY
    ("2% milk", "2% milk"),                             # fat content, not a quantity
]


@pytest.mark.parametrize("line,expected", ARTIFACT_CASES)
def test_known_parser_artifacts_stay_fixed(line, expected):
    """Each of these fragmented the ER working set into a string no USDA description contains.
    The worst, "salt and pepper to", reached 479 occurrences in the head stratum and resolved
    confidently to tomato chili sauce at 92 kcal/100g."""
    assert normalize(parse_line(line).ingredient_text) == expected


@pytest.mark.parametrize("line,_quantity,_unit,normalized", CASES)
def test_normalizing_an_already_normalized_string_changes_nothing(line, _quantity, _unit,
                                                                  normalized):
    """Normalization is a canonicalisation: the ER working set keys on it, the gold labels join
    on it, and `verify_sample` exists to catch strings orphaned by normalizer drift. It was NOT
    idempotent for 23 of 9,324 real strings — "package of chocolate chips" reduced to "of
    chocolate chip", "broccoli cuts" to "frozen broccoli cut" — because the strip passes ran in
    the wrong order. A non-idempotent normalizer leaves two spellings of one food as two
    entities, each separately resolved and separately wrong."""
    assert normalize(normalized) == normalized


# US recipe convention: capital T is tablespoon, lowercase t is teaspoon.
@pytest.mark.parametrize("line,unit", [
    ("3 T. butter", "tablespoon"),
    ("2 T minced garlic", "tablespoon"),
    ("1 T. olive oil", "tablespoon"),
    ("3 t. butter", "teaspoon"),
    ("1 t vanilla", "teaspoon"),
    ("2 tsp. sugar", "teaspoon"),
    ("2 tbsp. sugar", "tablespoon"),
])
def test_capital_t_is_tablespoon_and_lowercase_t_is_teaspoon(line, unit):
    """A 3x error. `UNITS` is keyed on the lowercased token and mapped "t" -> teaspoon, so every
    capital-T line was read as teaspoons — 23 lines in the corpus, against only 41 that use a
    lowercase t at all. Harmless while `unit` was unused, but Phase 3's gram conversion consumes
    it directly, so it would have understated those quantities by two thirds."""
    assert parse_line(line).unit == unit


def test_the_case_fix_does_not_touch_normalized_text():
    """`unit` and `ingredient_text` are split from the same token, and normalized_text is the
    join key for the entity map and all 300 gold labels. Changing which unit the token maps to
    must not change what is left over as the food."""
    assert parse_line("3 T. butter").ingredient_text == "butter"
    assert normalize(parse_line("3 T. butter").ingredient_text) == "butter"
    assert normalize(parse_line("3 t. butter").ingredient_text) == "butter"
