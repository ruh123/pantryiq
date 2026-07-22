"""Ingestion test: RecipeNLG → bronze.raw_recipes, lineage + idempotency (no dataset)."""
import csv

from pantryiq.ingestion.recipes import SCHEMA, ingest_recipes
from pantryiq.lakehouse.catalog import get_catalog, scan_with_duckdb


def _write_fixture(path, n=7):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["", "title", "ingredients", "directions", "link", "source", "NER"])
        for i in range(n):
            w.writerow(
                [
                    i,
                    f"Recipe {i}",
                    '["1 c. flour", "2 eggs"]',
                    '["mix", "bake"]',
                    f"http://example.com/{i}",
                    "Gathered",
                    '["flour", "eggs"]',
                ]
            )


def test_ingest_recipes_lineage_and_idempotent(tmp_path):
    csv_path = tmp_path / "recipes.csv"
    _write_fixture(csv_path, n=7)
    catalog = get_catalog(tmp_path / "lakehouse")

    table = ingest_recipes(catalog, csv_path=csv_path, limit=None)
    rows = table.scan().to_arrow()

    assert rows.num_rows == 7
    assert set(rows.column_names) == {field.name for field in SCHEMA}
    assert rows.column("source").to_pylist() == ["recipenlg"] * 7
    assert rows.column("recipe_id").to_pylist()[0] == "recipenlg:0"
    assert rows.column("source_url").to_pylist()[0] == "http://example.com/0"
    # DuckDB reads the same Bronze table
    assert scan_with_duckdb(catalog.load_table("bronze.raw_recipes")).num_rows == 7

    # Re-run must not duplicate rows.
    ingest_recipes(catalog, csv_path=csv_path, limit=None)
    assert catalog.load_table("bronze.raw_recipes").scan().to_arrow().num_rows == 7
