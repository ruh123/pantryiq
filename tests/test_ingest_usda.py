"""USDA ingestion test: build + write Bronze from injected foods (no network)."""
import pytest
import requests

from pantryiq.ingestion.usda import SCHEMA, _get_page, fetch_foods, ingest_usda_foods
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

    # Re-run with DIFFERENT input: the new data must WIN (1 row), not append (3) or no-op (2).
    ingest_usda_foods(catalog, foods=FOODS[:1])
    rows = catalog.load_table("bronze.raw_usda_foods").scan().to_arrow()
    assert rows.num_rows == 1
    assert rows.column("fdc_id").to_pylist() == ["167782"]


def test_empty_fetch_refuses_to_overwrite(tmp_path):
    """A 0-row fetch must never atomically replace a populated Bronze table."""
    catalog = get_catalog(tmp_path / "lakehouse")
    ingest_usda_foods(catalog, foods=FOODS)

    with pytest.raises(ValueError, match="0 rows"):
        ingest_usda_foods(catalog, foods=[])

    # The previous snapshot survives untouched.
    assert catalog.load_table("bronze.raw_usda_foods").scan().to_arrow().num_rows == 2


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        raise requests.HTTPError(f"HTTP {self.status_code}")


def _fake_get(responses, calls=None):
    def fake_get(url, params=None, timeout=None):
        if calls is not None:
            calls.append(params)
        return responses.pop(0)

    return fake_get


def test_get_page_retries_then_succeeds(monkeypatch):
    responses = [FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(200, [{"fdcId": 1}])]
    monkeypatch.setattr(requests, "get", _fake_get(responses))
    sleeps = []

    assert _get_page("k", "SR Legacy", 1, 200, sleeps.append) == [{"fdcId": 1}]
    assert sleeps == [3]  # Retry-After honored verbatim


def test_get_page_caps_absurd_retry_after(monkeypatch):
    responses = [FakeResponse(503, headers={"Retry-After": "999999"}), FakeResponse(200, [])]
    monkeypatch.setattr(requests, "get", _fake_get(responses))
    sleeps = []

    _get_page("k", "SR Legacy", 1, 200, sleeps.append)
    assert sleeps == [60]  # not ~11 days


def test_get_page_raises_on_non_retryable_status(monkeypatch):
    monkeypatch.setattr(requests, "get", _fake_get([FakeResponse(404)]))

    with pytest.raises(requests.HTTPError):
        _get_page("k", "SR Legacy", 1, 200, lambda _s: None)


def test_get_page_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(requests, "get", _fake_get([FakeResponse(503) for _ in range(5)]))

    with pytest.raises(RuntimeError, match="after 5 retries"):
        _get_page("k", "SR Legacy", 1, 200, lambda _s: None)


def test_get_page_rejects_non_list_body(monkeypatch):
    """A 200 carrying an error object must fail loudly, not iterate dict keys downstream."""
    monkeypatch.setattr(requests, "get", _fake_get([FakeResponse(200, {"error": "bad key"})]))

    with pytest.raises(RuntimeError, match="unexpected /foods/list payload"):
        _get_page("k", "SR Legacy", 1, 200, lambda _s: None)


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
