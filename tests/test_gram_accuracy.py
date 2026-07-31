"""Gram accuracy (3.3b): measuring whether conversions are RIGHT, not just present."""
import pytest

from pantryiq.silver.gram_accuracy import (
    REFERENCE_PATH,
    TOLERANCE,
    compare,
    load_reference,
    relative_error,
)


def test_relative_error_is_symmetric_and_bounded():
    """Same shape as `nutrition.error` on purpose — the two are read side by side, and a reader
    should not have to hold two definitions of "off by X%" at once."""
    assert relative_error(200.0, 200.0) == 0.0
    assert relative_error(110.0, 200.0) == pytest.approx(0.45)
    assert relative_error(200.0, 110.0) == pytest.approx(0.45)   # order must not matter
    assert relative_error(1.0, 1000.0) <= 1.0                    # bounded
    assert relative_error(0.0, 0.0) == 0.0                       # no division by zero


def test_a_conversion_inside_the_band_is_correct_and_outside_is_not():
    """Cooking references quote flour at 120-125 g/cup, so a band tighter than 10% would measure
    disagreement between sources rather than the pipeline."""
    reference = {("flour", "cup"): 125.0}

    assert compare(reference, {("flour", "cup"): (119.0, "volumetric", 10)})[0]["correct"]
    assert not compare(reference, {("flour", "cup"): (100.0, "volumetric", 10)})[0]["correct"]


def test_pairs_the_pipeline_never_converted_are_skipped_not_scored_as_wrong():
    """A reference pair the pipeline declined to convert is a COVERAGE gap (3.3), not an
    accuracy failure. Counting it here would double-penalise the same missing data and make the
    accuracy number move whenever coverage moved."""
    reference = {("flour", "cup"): 125.0, ("saffron", "pinch"): 0.1}

    rows = compare(reference, {("flour", "cup"): (125.0, "volumetric", 10)})

    assert [row["text"] for row in rows] == ["flour"]


def test_the_shipped_reference_table_is_well_formed():
    """Every row needs a source note: these values are the ground truth the whole measurement
    rests on, and unlike the ER labels a reader is expected to check them."""
    import csv

    with REFERENCE_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) >= 30
    for row in rows:
        assert float(row["grams_per_unit"]) > 0, row
        assert row["source_note"].strip(), row
        assert row["unit"].strip() and row["normalized_text"].strip(), row

    reference = load_reference()
    assert len(reference) == len(rows), "duplicate (text, unit) pair in the reference table"


def test_granulated_sugar_is_the_regression_this_module_exists_for():
    """The corpus's most frequent conversion (4,170 lines) reads 110 g/cup against a true 200 g,
    because bare "sugar" resolves to POWDERED sugar. §11's kcal metric could not see it — powdered
    and granulated sugar have near-identical energy per 100 g — and in mass terms it is a 45%
    error on the single most common line shape in the corpus.

    Pinned so that a future ER or portion change that fixes it shows up here as a win, and one
    that re-breaks it fails loudly."""
    reference = load_reference()

    assert reference[("sugar", "cup")] == 200.0
    assert relative_error(110.0, reference[("sugar", "cup")]) > TOLERANCE
