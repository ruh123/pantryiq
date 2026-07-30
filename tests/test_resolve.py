"""The shipped resolver: one pick per string, by cosine, with a stored confidence curve."""
import json

import duckdb
import pytest

from pantryiq.er.resolve import LOW_CONFIDENCE, load_curve, resolve, save_curve, top_picks


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA silver")
    con.execute("CREATE TABLE silver.ingredient_candidates "
                "(normalized_text VARCHAR, fdc_id VARCHAR, cosine_search DOUBLE)")
    for text, fdc_id, cosine in [
        ("egg", "1", 0.70), ("egg", "2", 0.95), ("egg", "3", 0.60),
        ("flour", "4", 0.42), ("flour", "5", 0.41),
    ]:
        con.execute("INSERT INTO silver.ingredient_candidates VALUES (?,?,?)",
                    [text, fdc_id, cosine])
    con.close()
    return path


@pytest.fixture
def curve(tmp_path):
    path = tmp_path / "curve.json"
    path.write_text(json.dumps({"x": [0.40, 0.60, 0.95], "y": [0.10, 0.50, 0.90],
                                "equivalent_within": 0.10, "fitted_on": "test"}))
    return path


def test_the_pick_is_the_highest_cosine_not_the_first_row(db):
    """0.95 sits second in insertion order — a resolver reading table order would take 0.70."""
    picks = dict((text, fdc_id) for text, fdc_id, _ in top_picks(db))

    assert picks["egg"] == "2"


def test_exactly_one_pick_per_string(db):
    picks = top_picks(db)

    assert len(picks) == len({text for text, *_ in picks}) == 2


def test_output_is_stable_across_identical_calls(db):
    """Found by this test: without an outer ORDER BY, DuckDB returned the same picks in a
    different row order each call, making run-to-run diffs of the output pure noise."""
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO silver.ingredient_candidates VALUES ('salt','9',0.5),('salt','8',0.5)")
    con.close()

    assert top_picks(db) == top_picks(db)
    assert dict((t, f) for t, f, _ in top_picks(db))["salt"] == "8"  # lowest fdc_id


def test_restricting_to_a_string_list(db):
    assert {text for text, *_ in top_picks(db, ["flour"])} == {"flour"}


def test_the_stored_curve_maps_cosine_to_confidence(curve):
    confidence = load_curve(curve)

    assert confidence(0.95) == pytest.approx(0.90)
    assert 0.10 < confidence(0.50) < 0.50   # interpolated between stored points
    assert confidence(0.20) == pytest.approx(0.10)  # clipped below the fitted range
    assert confidence(1.00) == pytest.approx(0.90)  # clipped above


def test_resolving_flags_low_confidence_picks(db, curve):
    rows = {text: (fdc_id, confidence, flagged)
            for text, fdc_id, _, confidence, flagged, _ in resolve(db, curve, abstain_threshold=0.0)}

    assert rows["egg"][2] is False       # confidence 0.90
    assert rows["flour"][2] is True      # cosine 0.42 -> well under the cut
    assert rows["flour"][1] < LOW_CONFIDENCE


def test_abstention_is_a_separate_signal_from_the_confidence_flag(db, curve):
    """They answer different questions: `flagged` means "this pick may have the wrong facet",
    `abstained` means "this string probably has no USDA entity at all". Raw cosine separates the
    null class better than the nutrition-fitted curve, so abstention does not reuse it."""
    rows = {text: (flagged, abstained)
            for text, _, _, _, flagged, abstained in resolve(db, curve, abstain_threshold=0.60)}

    # egg: cosine 0.95 -> confident AND not abstained
    assert rows["egg"] == (False, False)
    # flour: cosine 0.42 -> below both cuts
    assert rows["flour"] == (True, True)


def test_a_zero_threshold_never_abstains(db, curve):
    """The stored threshold is absent until abstain.py has been run; the resolver must then
    behave exactly as it did before rather than declining on everything."""
    rows = resolve(db, curve, abstain_threshold=0.0)

    assert not any(abstained for *_, abstained in rows)


def test_the_curve_round_trips_through_disk(tmp_path):
    """Resolving must be a lookup, not a fit — the stored curve is the interface."""
    from sklearn.isotonic import IsotonicRegression

    fitted = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        [0.4, 0.6, 0.8, 1.0], [0.0, 0.0, 1.0, 1.0])
    path = save_curve(fitted, tmp_path / "c.json")

    confidence = load_curve(path)
    assert confidence(0.4) < confidence(1.0)
