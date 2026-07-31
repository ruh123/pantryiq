"""The curated cost table. Prices are a stand-in; the table's integrity is not."""
import csv
from pathlib import Path

import pytest

COST_PATH = Path("data/cost_reference.csv")


def _rows():
    with COST_PATH.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_every_row_has_a_positive_price_and_an_entity():
    """A zero or missing price would silently contribute $0 to a recipe total while still
    counting as "costed" — the same null-vs-zero failure the gram layer guards against."""
    rows = _rows()

    assert len(rows) >= 90
    for row in rows:
        assert row["fdc_id"].strip().isdigit(), row
        assert float(row["usd_per_kg"]) > 0, row
        assert row["description"].strip(), row


def test_no_entity_is_priced_twice():
    """A duplicate would fan out the join in `recipe_nutrition` and double-count that
    ingredient's cost."""
    ids = [row["fdc_id"] for row in _rows()]

    assert len(ids) == len(set(ids))


def test_the_csv_survives_a_real_parser():
    """Found by this test: a note containing an unquoted comma made one row 5 fields wide, and
    DuckDB's delimiter sniffer then read the ENTIRE header as a single column — the model failed
    to build with "column fdc_id not found" while the file looked fine to the eye."""
    for row in _rows():
        assert set(row) == {"fdc_id", "usd_per_kg", "description", "note"}, row
        assert None not in row.values(), f"ragged row: {row}"


@pytest.mark.skipif(not Path("data/pantryiq.duckdb").exists(),
                    reason="corpus not present")
def test_every_priced_entity_exists_in_the_canonical_set():
    """A price for an fdc_id that is not in `silver.usda_foods` joins to nothing and is silent
    dead weight — it would look like coverage while contributing none."""
    import duckdb

    con = duckdb.connect("data/pantryiq.duckdb", read_only=True)
    try:
        real = {row[0] for row in con.execute(
            "SELECT fdc_id FROM silver.usda_foods").fetchall()}
    finally:
        con.close()

    unknown = {row["fdc_id"] for row in _rows()} - real
    assert not unknown, f"priced entities missing from usda_foods: {sorted(unknown)[:5]}"
