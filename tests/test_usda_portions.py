"""Silver food portions: grams per measure, and the three source quirks that corrupt them."""
import duckdb
import pyarrow as pa
import pytest

from pantryiq.silver.usda_portions import (
    build_rows,
    grams_per_unit,
    parse_measure,
    write_silver,
)

# (modifier, measureUnit.name, measure, qualifier) — every case is a real USDA portion shape.
CASES = [
    # SR Legacy: measureUnit is the literal "undetermined", the measure lives in modifier.
    ("cup", "undetermined", "cup", None),
    ("tbsp", "undetermined", "tablespoon", None),
    ("tsp", "undetermined", "teaspoon", None),
    ("oz", "undetermined", "ounce", None),
    ("slice", "undetermined", "slice", None),
    # Comma-separated qualifiers — worth 25% on flour, so they are kept, not discarded.
    ("cup, chopped", "undetermined", "cup", "chopped"),
    ("cup, sifted", "undetermined", "cup", "sifted"),
    ("cup, melted", "undetermined", "cup", "melted"),
    # Space-separated qualifiers, the same thing written differently.
    ("cup packed", "undetermined", "cup", "packed"),
    ("cup chopped", "undetermined", "cup", "chopped"),
    # Volume, NOT mass — must not collapse onto "ounce".
    ("fl oz", "undetermined", "fluid_ounce", None),
    # Parentheticals describe the measure; they are not part of it.
    ("oz (23 whole kernels)", "undetermined", "ounce", "23 whole kernels"),
    ("can (10.75 oz)", "undetermined", "can", "10.75 oz"),
    ("cubic inch", "undetermined", "cubic_inch", None),
    # Foundation: modifier is null and measureUnit carries the measure.
    (None, "milliliter", "milliliter", None),
    (None, "RACC", "racc", None),
    # Nothing usable either side.
    (None, "undetermined", None, None),
    (None, None, None, None),
    ("", "", None, None),
]


@pytest.mark.parametrize("modifier,measure_unit,measure,qualifier", CASES)
def test_parse_measure(modifier, measure_unit, measure, qualifier):
    assert parse_measure(modifier, measure_unit) == (measure, qualifier)


def test_the_modifier_wins_over_the_measure_unit():
    """95% of the canonical set is SR Legacy, where measureUnit.name is the string
    "undetermined" and the real measure is in modifier. Reading measureUnit first would discard
    nearly every usable row in the table."""
    assert parse_measure("cup", "undetermined") == ("cup", None)
    assert parse_measure("tbsp", "gram") == ("tablespoon", None)


def test_a_fluid_ounce_is_not_an_ounce():
    """The recipe-side UNITS map sends the bare token "oz" to `ounce` (mass). USDA's "fl oz" is
    a volume, and conflating them would silently price every fluid-ounce portion at 28.35g.
    Keeping them distinct means the join simply misses — an honest gap, per guide rule 4."""
    assert parse_measure("fl oz", "undetermined")[0] == "fluid_ounce"
    assert parse_measure("oz", "undetermined")[0] == "ounce"
    assert parse_measure("fl oz", None)[0] != parse_measure("oz", None)[0]


def test_gram_weight_is_divided_by_the_amount():
    """A row reading amount=2, gramWeight=28.4 means TWO tbsp weigh 28.4g. Taking gramWeight at
    face value doubles every such food's contribution to a recipe."""
    assert grams_per_unit({"amount": 2.0, "gramWeight": 28.4}) == pytest.approx(14.2)
    assert grams_per_unit({"amount": 1.0, "gramWeight": 240.0}) == 240.0
    assert grams_per_unit({"gramWeight": 240.0}) == 240.0        # absent amount means one


@pytest.mark.parametrize("portion", [
    {"amount": 0, "gramWeight": 28.4},        # would be a ZeroDivisionError
    {"amount": -1.0, "gramWeight": 28.4},
    {"amount": 1.0, "gramWeight": 0},         # zero grams is not a weight
    {"amount": 1.0, "gramWeight": None},
    {"amount": 1.0},
])
def test_an_unusable_weight_is_none_not_a_number(portion):
    """A null is a measured gap; a zero is a claim that the ingredient contributes nothing. The
    project's whole thesis is that those must never be confused."""
    assert grams_per_unit(portion) is None


class _FakeBronze:
    def __init__(self, rows):
        self._rows = rows

    def load_table(self, name):
        assert name == "bronze.raw_usda_portions"
        return self

    def scan(self):
        return self

    def to_arrow(self):
        return pa.table({
            "fdc_id": [row[0] for row in self._rows],
            "raw_payload": [row[1] for row in self._rows],
        })


def test_build_rows_skips_portions_it_cannot_use():
    """A portion with no parseable measure or no usable weight carries no information; landing
    it as a row with NULLs would let a downstream join produce a silent zero."""
    table = build_rows(_FakeBronze([
        ("10", '[{"measureUnit": {"name": "undetermined"}, "modifier": "cup", '
               '"amount": 1, "gramWeight": 240.0}]'),
        ("11", '[{"measureUnit": {"name": "undetermined"}, "modifier": null, "gramWeight": 5.0}]'),
        ("12", '[{"measureUnit": {"name": "undetermined"}, "modifier": "cup", "gramWeight": 0}]'),
        ("13", "[]"),
    ]))

    assert table.column("fdc_id").to_pylist() == ["10"]
    assert table.column("gram_weight").to_pylist() == [240.0]


def test_duplicate_measures_collapse_to_the_median(tmp_path):
    """USDA lists the same measure more than once for some foods. A mean is dragged by one
    outlier; the median is not, and a join needs exactly one weight per (food, measure)."""
    rows = pa.table({
        "fdc_id": ["10", "10", "10", "10"],
        "measure": ["cup", "cup", "cup", "tablespoon"],
        "qualifier": [None, None, None, None],
        "gram_weight": [120.0, 130.0, 900.0, 15.0],
    })

    db = write_silver(rows, tmp_path / "t.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    try:
        result = dict(con.execute(
            "SELECT measure, gram_weight FROM silver.usda_portions").fetchall())
    finally:
        con.close()

    assert result["cup"] == 130.0        # median of (120, 130, 900), not the 383.3 mean
    assert result["tablespoon"] == 15.0


def test_a_qualified_measure_stays_a_separate_row(tmp_path):
    """Sifted flour is 100g/cup against 125g unsifted. Collapsing the qualifier would pick one
    silently; keeping both rows leaves the choice — and its sensitivity — measurable."""
    rows = pa.table({
        "fdc_id": ["10", "10"],
        "measure": ["cup", "cup"],
        "qualifier": [None, "sifted"],
        "gram_weight": [125.0, 100.0],
    })

    db = write_silver(rows, tmp_path / "t.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    try:
        result = con.execute("SELECT qualifier, gram_weight FROM silver.usda_portions "
                             "ORDER BY qualifier NULLS FIRST").fetchall()
    finally:
        con.close()

    assert result == [(None, 125.0), ("sifted", 100.0)]
