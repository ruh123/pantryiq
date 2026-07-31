"""Gram conversion: the four paths, and the nulls that must never become zeros."""
import duckdb
import pytest

from pantryiq.silver.grams import (
    convert,
    from_count,
    from_mass,
    from_pack_size,
    from_volume,
    write_silver,
)

# A food with a full set of volume portions, plus a per-item one.
FLOUR = {"cup": {None: 125.0, "sifted": 100.0}, "tablespoon": {None: 7.8}}
EGG = {"large": {None: 50.0}}
MILK = {"cup": {None: 244.0}}
NOTHING: dict = {}


# (quantity, unit, line, portions, grams, method) — every line is a real corpus shape.
CASES = [
    ("1 lb. ground beef",            1.0, "pound", NOTHING, 453.59237, "mass"),
    ("8 oz. cream cheese",           8.0, "ounce", NOTHING, 226.796185, "mass"),
    ("500 g flour",                  500.0, "gram", NOTHING, 500.0, "mass"),
    # Pack size: the parser keeps quantity+unit and drops the only part stating a mass.
    ("2 (16 oz.) pkg. frozen corn",  2.0, "package", NOTHING, 907.184744, "pack_size"),
    ("1 (10 1/2 oz.) can soup",      1.0, "can", NOTHING, 297.669992625, "pack_size"),
    # Volumetric, against the entity's own cup weight.
    ("2 c. flour",                   2.0, "cup", FLOUR, 250.0, "volumetric"),
    ("1 tbsp. flour",                1.0, "tablespoon", FLOUR, 7.8, "volumetric"),
    # A quart of milk against a food that only publishes a cup: exact volume ratio.
    ("1 qt. milk",                   1.0, "quart", MILK, 976.0, "volumetric_derived"),
    # Count: no unit at all.
    ("1 egg",                        1.0, None, EGG, 50.0, "count"),
    ("3 eggs",                       3.0, None, EGG, 150.0, "count"),
]


@pytest.mark.parametrize("line,quantity,unit,portions,grams,method", CASES)
def test_convert(line, quantity, unit, portions, grams, method):
    result = convert(quantity, unit, line, portions)

    assert result is not None, f"{line!r} did not convert"
    assert result[0] == pytest.approx(grams, rel=1e-6)
    assert result[1] == method


@pytest.mark.parametrize("quantity,unit,line,portions", [
    (None, "cup", "flour to taste", FLOUR),      # no quantity at all
    (2.0, "cup", "2 c. mystery", NOTHING),       # no portions for this food
    (1.0, "package", "1 pkg. mystery", NOTHING),  # a container with no stated size
    (1.0, None, "1 whole chicken", NOTHING),     # count with no per-item portion
    (2.0, "clove", "2 cloves garlic", MILK),     # unit is neither mass, volume, nor countable
])
def test_an_unconvertible_line_is_none_not_zero(quantity, unit, line, portions):
    """A zero claims the ingredient weighs nothing and would drag a recipe's totals down while
    looking like a real measurement. A null is a measured gap that `nutrition_coverage` reports.
    This is the same discipline as `nutrition.error` returning None for a missing kcal."""
    assert convert(quantity, unit, line, portions) is None


def test_mass_needs_no_portion_data_at_all():
    """A pound is a pound whether or not USDA published a portion for the food — and whether or
    not the resolver even identified it. This path is why declined strings still get some
    coverage."""
    assert from_mass(1.0, "pound") == (pytest.approx(453.59237), "mass")
    assert from_mass(1.0, "cup") is None


def test_the_foods_own_weight_for_the_measure_beats_deriving_one():
    """Found by this test: `from_volume` searched the volume family largest-first and priced
    "1 tbsp flour" off the CUP weight — 125/16 = 7.81g — while USDA publishes 7.8g for the
    tablespoon directly. The ratios between measures are exact, but the weights are separately
    measured and things pack differently at different scales, so a derived weight is an
    approximation standing in front of a measurement."""
    assert from_volume(1.0, "tablespoon", FLOUR) == (7.8, "volumetric")
    assert from_volume(1.0, "cup", FLOUR) == (125.0, "volumetric")

    # Only when the exact measure is absent is one derived — and it is labelled as derived.
    assert from_volume(1.0, "quart", MILK) == (pytest.approx(976.0), "volumetric_derived")


def test_the_unqualified_portion_wins_over_a_qualified_one():
    """USDA carries "cup" at 125g and "cup, sifted" at 100g for flour, and "1 cup flour" has not
    said which. Taking the plain measure is the stated assumption; picking the qualified variant
    would silently apply a 20% discount."""
    assert from_volume(1.0, "cup", FLOUR) == (125.0, "volumetric")


def test_only_qualified_portions_fall_back_to_their_median():
    """With no unqualified row there is no truth to prefer, and an arbitrary pick would be a
    coin flip between "chopped" and "melted"."""
    only_qualified = {"cup": {"chopped": 100.0, "melted": 200.0, "packed": 180.0}}

    assert from_volume(1.0, "cup", only_qualified) == (180.0, "volumetric")


def test_a_volume_pack_size_routes_through_density_not_mass():
    """"1 (12 fl oz) can" states a VOLUME. Treating 12 fl oz as 12 ounces of mass would be a
    density assumption of exactly 1.0 g/ml applied silently to every food."""
    assert from_pack_size(1.0, "1 (12 fl oz) can evaporated milk", NOTHING) is None

    result = from_pack_size(1.0, "1 (500 ml) carton milk", MILK)
    assert result is not None and result[1] == "pack_size_volume"
    assert result[0] == pytest.approx(500 / 236.5882365 * 244.0, rel=1e-6)


def test_mass_is_preferred_over_a_pack_size_when_both_are_present():
    """"1 lb. (16 oz.) bag" states the same thing twice; the parsed unit is the more direct
    reading and must not be double-counted against the parenthetical."""
    result = convert(1.0, "pound", "1 lb. (16 oz.) bag of sugar", FLOUR)

    assert result == (pytest.approx(453.59237), "mass")


def test_a_count_line_does_not_borrow_a_volume_weight():
    """"1 onion" must not be priced as one cup of onion. Only a per-item portion can price a
    count, or the line stays null."""
    assert from_count(1.0, None, MILK) is None
    assert from_count(1.0, None, EGG) == (50.0, "count")


def test_a_unit_bearing_line_never_falls_through_to_count():
    """`from_count` fires only when the line named no unit; "2 cups" with no cup portion is a
    gap, not an invitation to price it per item."""
    assert from_count(2.0, "cup", EGG) is None


def test_written_rows_round_trip_with_nulls_intact(tmp_path):
    """The null must survive the write as a NULL, not as a 0.0 — a silently-zeroed column is
    exactly the failure this table's `converted` flag exists to make visible."""
    import pyarrow as pa

    rows = pa.table({
        "recipe_id": ["r1", "r1"],
        "line_index": [0, 1],
        "grams": [125.0, None],
        "method": ["volumetric", None],
        "converted": [True, False],
    })

    db = write_silver(rows, tmp_path / "t.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    try:
        result = con.execute("SELECT grams, method, converted FROM "
                             "silver.recipe_ingredient_grams ORDER BY line_index").fetchall()
    finally:
        con.close()

    assert result == [(125.0, "volumetric", True), (None, None, False)]
