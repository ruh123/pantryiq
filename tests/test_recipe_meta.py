"""Recipe titles into Silver — including the snapshot trap that doubles every row."""
import json

import duckdb
import pyarrow as pa
import pytest

from pantryiq.silver.recipe_meta import build_rows, write_silver


class _FakeBronze:
    """Duck-types the catalog, and `scan_with_duckdb` is patched to return this table."""

    def __init__(self, rows):
        self._rows = rows

    def load_table(self, name):
        assert name == "bronze.raw_recipes"
        return self

    def to_arrow(self):
        return pa.table({
            "recipe_id": [row[0] for row in self._rows],
            "raw_payload": [row[1] for row in self._rows],
        })


@pytest.fixture
def patched_scan(monkeypatch):
    """`build_rows` calls scan_with_duckdb, which needs a real Iceberg table — stub it."""
    monkeypatch.setattr("pantryiq.lakehouse.catalog.scan_with_duckdb",
                        lambda table, con=None: table.to_arrow())


def _payload(**fields):
    return json.dumps({"title": "T", "link": "example.com/1", **fields})


def test_titles_and_urls_are_lifted_from_the_payload(patched_scan):
    rows = build_rows(_FakeBronze([
        ("recipenlg:0", _payload(title="No-Bake Nut Cookies", link="www.cookbooks.com/x")),
    ]))

    assert rows.column("title").to_pylist() == ["No-Bake Nut Cookies"]
    assert rows.column("source_url").to_pylist() == ["www.cookbooks.com/x"]


def test_trailing_whitespace_is_stripped_from_titles(patched_scan):
    """The corpus ships titles like 'Reeses Cups(Candy)  ' — that whitespace would otherwise
    reach a user verbatim inside an agent's answer."""
    rows = build_rows(_FakeBronze([("recipenlg:4", _payload(title="Reeses Cups(Candy)  "))]))

    assert rows.column("title").to_pylist() == ["Reeses Cups(Candy)"]


def test_an_absent_title_is_null_not_an_empty_string(patched_scan):
    """A recipe with no title has no title. An empty string reads as a real value downstream and
    would render as a blank name rather than an obvious gap."""
    rows = build_rows(_FakeBronze([
        ("recipenlg:1", _payload(title="", link="")),
        ("recipenlg:2", json.dumps({"directions": "..."})),   # neither key present at all
        ("recipenlg:3", _payload(title="   ")),               # whitespace-only is also absent
    ]))

    assert rows.column("title").to_pylist() == [None, None, None]
    assert rows.column("source_url").to_pylist() == [None, None, "example.com/1"]


def test_the_scheme_less_url_is_preserved_as_written(patched_scan):
    """Payload links have no scheme ('www.cookbooks.com/...'). Prepending https:// would assert
    something about a site nobody checked; Bronze is source-preserving and Silver keeps that."""
    rows = build_rows(_FakeBronze([("recipenlg:0", _payload(link="www.cookbooks.com/x"))]))

    assert rows.column("source_url").to_pylist() == ["www.cookbooks.com/x"]


def test_rows_round_trip_to_duckdb(tmp_path, patched_scan):
    rows = build_rows(_FakeBronze([
        ("recipenlg:0", _payload(title="A")),
        ("recipenlg:1", _payload(title="B")),
    ]))

    db = write_silver(rows, tmp_path / "t.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    try:
        result = con.execute(
            "SELECT recipe_id, title FROM silver.recipe_meta ORDER BY recipe_id").fetchall()
    finally:
        con.close()

    assert result == [("recipenlg:0", "A"), ("recipenlg:1", "B")]


@pytest.mark.skipif(not __import__("pathlib").Path("data/lakehouse").exists(),
                    reason="lakehouse not present")
def test_the_bronze_read_uses_the_iceberg_snapshot_not_a_parquet_glob():
    """The trap this module exists to avoid: `raw_recipes/data/` holds TWO parquet files of
    15,000 rows each — an orphan from a DELETE+APPEND — with identical recipe_ids. A glob returns
    30,000 and silently doubles every downstream join. Measured here so a future 'simplification'
    to read_parquet fails loudly."""
    from pantryiq.lakehouse.catalog import get_catalog, scan_with_duckdb

    via_iceberg = scan_with_duckdb(get_catalog().load_table("bronze.raw_recipes")).num_rows
    via_glob = duckdb.connect().execute(
        "SELECT count(*) FROM read_parquet('data/lakehouse/bronze/raw_recipes/data/*.parquet')"
    ).fetchone()[0]

    assert via_iceberg == 15_000
    assert via_glob > via_iceberg, "the orphan snapshot is gone — re-check this guard"
