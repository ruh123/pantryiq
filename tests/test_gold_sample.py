"""Gold sampling: stratification, determinism, and the frozen holdout split."""
import duckdb
import pytest

from pantryiq.er.gold import (
    SAMPLE_PER_STRATUM,
    TUNE_PER_STRATUM,
    draw_sample,
    holdout_fingerprint,
    write_sample,
)


@pytest.fixture
def db(tmp_path):
    """A miniature Silver with enough strings per stratum to draw the real sample sizes."""
    path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA silver")
    con.execute(
        "CREATE TABLE silver.distinct_ingredient_strings "
        "(normalized_text VARCHAR, occurrence_count INTEGER, frequency_stratum VARCHAR)"
    )
    con.execute(
        "CREATE TABLE silver.recipe_ingredient_lines (normalized_text VARCHAR, line_raw VARCHAR)"
    )
    for stratum, count in (("head", 50), ("mid", 10), ("tail", 1)):
        for i in range(SAMPLE_PER_STRATUM[stratum] + 30):
            con.execute("INSERT INTO silver.distinct_ingredient_strings VALUES (?, ?, ?)",
                        [f"{stratum}-food-{i}", count, stratum])
            con.execute("INSERT INTO silver.recipe_ingredient_lines VALUES (?, ?)",
                        [f"{stratum}-food-{i}", f"1 c. {stratum}-food-{i}"])
    con.close()
    return path


def test_sample_is_stratified_and_split(db):
    rows = draw_sample(db)

    assert len(rows) == 500
    for stratum, size in SAMPLE_PER_STRATUM.items():
        in_stratum = [r for r in rows if r["frequency_stratum"] == stratum]
        assert len(in_stratum) == size
        assert sum(r["split"] == "tune" for r in in_stratum) == TUNE_PER_STRATUM[stratum]
    assert sum(r["split"] == "tune" for r in rows) == 300
    assert sum(r["split"] == "holdout" for r in rows) == 200


def test_sample_is_deterministic(db):
    """A re-draw must reproduce the sample exactly, or the frozen holdout means nothing."""
    first, second = draw_sample(db), draw_sample(db)

    assert [r["normalized_text"] for r in first] == [r["normalized_text"] for r in second]
    assert holdout_fingerprint(first) == holdout_fingerprint(second)


def test_seed_change_moves_the_sample(db):
    assert holdout_fingerprint(draw_sample(db)) != holdout_fingerprint(draw_sample(db, seed=1))


def test_no_string_is_in_both_splits(db):
    rows = draw_sample(db)
    tune = {r["normalized_text"] for r in rows if r["split"] == "tune"}
    holdout = {r["normalized_text"] for r in rows if r["split"] == "holdout"}

    assert not tune & holdout
    assert len(tune) + len(holdout) == 500  # and no duplicates within the sample


def test_rows_carry_example_lines(db):
    rows = draw_sample(db)
    assert all(r["example_lines"] for r in rows)


def test_write_sample_records_the_fingerprint(db, tmp_path):
    import json

    rows = draw_sample(db)
    path = write_sample(rows, tmp_path / "gold")

    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 500

    manifest = json.loads((tmp_path / "gold" / "manifest.json").read_text())
    assert manifest["tune"] == 300
    assert manifest["holdout"] == 200
    assert manifest["holdout_fingerprint"] == holdout_fingerprint(rows)


def test_refuses_a_stratum_that_is_too_small(tmp_path):
    path = tmp_path / "small.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA silver")
    con.execute(
        "CREATE TABLE silver.distinct_ingredient_strings "
        "(normalized_text VARCHAR, occurrence_count INTEGER, frequency_stratum VARCHAR)"
    )
    con.execute("CREATE TABLE silver.recipe_ingredient_lines (normalized_text VARCHAR, line_raw VARCHAR)")
    con.execute("INSERT INTO silver.distinct_ingredient_strings VALUES ('salt', 99, 'head')")
    con.close()

    with pytest.raises(ValueError, match="need 167"):
        draw_sample(path)
