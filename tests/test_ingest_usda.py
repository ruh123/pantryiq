"""USDA ingestion test: build + write Bronze from injected foods (no network)."""
from pantryiq.ingestion.usda import SCHEMA, fetch_foods, ingest_usda_foods
from pantryiq.lakehouse.catalog import get_catalog, scan_with_duckdb

FOODS = [
    {
        "fdcId": 167782,
        "dataType": "SR Legacy",
        "description": "Abiyuch, raw",
        "foodNutrients": [{"name": "Energy", "amount": 69.0, "unitName": "KCAL"}],
    },
    {
        "fdcId": 1105,
        "dataType": "Foundation",
        "description": "Cheese, cheddar",
        "foodNutrients": [],
    },
]


def test_usda_bronze_lineage_and_idempotent(tmp_path):
    catalog = get_catalog(tmp_path / "lakehouse")

    table = ingest_usda_foods(catalog, foods=FOODS)
    rows = table.scan().to_arrow()

    assert rows.num_rows == 2
    assert set(rows.column_names) == {field.name for field in SCHEMA}
    assert rows.column("fdc_id").to_pylist() == ["167782", "1105"]
    assert rows.column("source").to_pylist() == ["usda_fdc", "usda_fdc"]
    assert set(rows.column("data_type").to_pylist()) == {"SR Legacy", "Foundation"}
    assert scan_with_duckdb(catalog.load_table("bronze.raw_usda_foods")).num_rows == 2

    # Re-run must not duplicate rows.
    ingest_usda_foods(catalog, foods=FOODS)
    assert catalog.load_table("bronze.raw_usda_foods").scan().to_arrow().num_rows == 2


def test_fetch_foods_paginates_and_stops(monkeypatch):
    # Two full pages then a short page => generator yields 5 and stops (no infinite loop).
    pages = {1: [{"fdcId": i} for i in range(2)], 2: [{"fdcId": i} for i in range(2)], 3: [{"fdcId": 99}]}
    calls = []

    def fake_get_page(api_key, data_type, page, page_size, sleep):
        calls.append((data_type, page))
        return pages.get(page, [])

    monkeypatch.setattr("pantryiq.ingestion.usda._get_page", fake_get_page)
    got = list(fetch_foods("k", data_types=("SR Legacy",), page_size=2, sleep=lambda _s: None))
    assert len(got) == 5
    assert calls == [("SR Legacy", 1), ("SR Legacy", 2), ("SR Legacy", 3)]
