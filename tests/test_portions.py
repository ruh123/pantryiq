"""Bronze food portions: the gram weights Phase 3 needs, and the batching/retry around them."""
import json

import pytest
import requests

from pantryiq.ingestion.portions import (
    BATCH_SIZE,
    TABLE,
    _post_batch,
    build_bronze_table,
    fetch_portions,
    ingest_usda_portions,
    load_cache,
)
from pantryiq.lakehouse.catalog import get_catalog, scan_with_duckdb


class _Response:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code}")


def _portion(measure, modifier, grams):
    return {"measureUnit": {"name": measure}, "modifier": modifier, "gramWeight": grams}


def test_a_batch_is_keyed_back_to_the_ids_that_were_asked_for(monkeypatch):
    """The API returns foods in its own order and may omit ids entirely. Zipping the response
    positionally against the request would attach one food's gram weights to another food —
    silently wrong nutrition, which is the failure this project exists to prevent."""
    monkeypatch.setattr("pantryiq.ingestion.portions._post_batch",
                        lambda key, ids, sleep: [
                            {"fdcId": 20, "foodPortions": [_portion("undetermined", "cup", 240.0)]},
                            {"fdcId": 10, "foodPortions": [_portion("undetermined", "tsp", 4.2)]},
                        ])

    result = dict(fetch_portions("key", ["10", "20"], cache_path=None))

    assert result["10"][0]["gramWeight"] == 4.2
    assert result["20"][0]["gramWeight"] == 240.0


def test_a_food_with_no_portions_is_recorded_as_empty_not_skipped(monkeypatch):
    """"USDA has none" and "we never asked" are different facts, and only the first can be
    verified later. Dropping the row would make the two indistinguishable downstream."""
    monkeypatch.setattr("pantryiq.ingestion.portions._post_batch",
                        lambda key, ids, sleep: [{"fdcId": 10, "foodPortions": []}])

    result = dict(fetch_portions("key", ["10", "11"], cache_path=None))

    assert result == {"10": [], "11": []}   # 11 was omitted by the API entirely


def test_ids_are_split_into_batches_of_twenty(monkeypatch):
    """The endpoint caps at 20 ids per POST; a 21-id request silently returns a partial set."""
    seen = []

    def fake_post(key, ids, sleep):
        seen.append(len(ids))
        return [{"fdcId": int(i), "foodPortions": []} for i in ids]

    monkeypatch.setattr("pantryiq.ingestion.portions._post_batch", fake_post)
    list(fetch_portions("key", [str(n) for n in range(45)], cache_path=None))

    assert seen == [20, 20, 5]
    assert all(size <= BATCH_SIZE for size in seen)


def test_post_batch_retries_a_rate_limit_then_succeeds(monkeypatch):
    """The riskiest code in any ingestion module is the part that talks to the network, and the
    sibling foods pull shipped with it untested. 429 must retry, not raise."""
    responses = [_Response(429, headers={"Retry-After": "2"}),
                 _Response(200, [{"fdcId": 1, "foodPortions": []}])]
    slept = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: responses.pop(0))

    result = _post_batch("key", ["1"], slept.append)

    assert result == [{"fdcId": 1, "foodPortions": []}]
    assert slept == [2]


def test_post_batch_caps_an_absurd_retry_after(monkeypatch):
    """A Retry-After of 999999 would stall the pull for ~11 days."""
    responses = [_Response(503, headers={"Retry-After": "999999"}), _Response(200, [])]
    slept = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: responses.pop(0))

    _post_batch("key", ["1"], slept.append)

    assert slept == [60]


def test_post_batch_rejects_a_two_hundred_that_is_not_a_list(monkeypatch):
    """An error body served with HTTP 200 would otherwise iterate dict keys and produce
    nonsense rows rather than failing."""
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Response(200, {"error": "over rate limit"}))

    with pytest.raises(RuntimeError, match="not a list"):
        _post_batch("key", ["1"], lambda _: None)


def test_post_batch_gives_up_after_exhausting_retries(monkeypatch):
    """Failing loudly beats landing a partial table that looks complete."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Response(503))

    with pytest.raises(RuntimeError, match="after 5 retries"):
        _post_batch("key", ["1"], lambda _: None)


def test_an_empty_result_refuses_to_overwrite(tmp_path):
    """Iceberg's unfiltered overwrite atomically replaces the table, so a 0-row pull would wipe
    it with no error at all — the P0 caught on the foods pull, guarded here from the start."""
    catalog = get_catalog(tmp_path / "warehouse")

    with pytest.raises(ValueError, match="0 rows"):
        ingest_usda_portions(catalog, records=[])


def test_lineage_and_idempotent_replacement(tmp_path):
    """A re-run with DIFFERENT input must replace rather than append or no-op — asserting only
    a row count against an identical re-run cannot tell those apart."""
    catalog = get_catalog(tmp_path / "warehouse")

    first = ingest_usda_portions(catalog, records=[
        ("10", [_portion("undetermined", "cup", 240.0)]),
        ("11", []),
        ("12", [_portion("undetermined", "tsp", 4.2)]),
    ])
    assert first.scan().to_arrow().num_rows == 3

    second = ingest_usda_portions(catalog, records=[("10", [_portion("g", None, 1.0)])])
    rows = second.scan().to_arrow()

    assert rows.num_rows == 1                                  # 1 = replaced (3 = no-op, 4 = append)
    assert rows.column("fdc_id").to_pylist() == ["10"]
    assert rows.column("source").to_pylist() == ["usda_fdc"]   # lineage preserved
    assert json.loads(rows.column("raw_payload")[0].as_py())[0]["gramWeight"] == 1.0


def test_duckdb_reads_the_portions_back(tmp_path):
    """Bronze is only useful if Silver can read it — the round-trip dbt/DuckDB will rely on."""
    catalog = get_catalog(tmp_path / "warehouse")
    ingest_usda_portions(catalog, records=[("10", [_portion("undetermined", "cup", 240.0)])])

    table = scan_with_duckdb(catalog.load_table(TABLE))

    assert table.num_rows == 1
    payload = json.loads(table.column("raw_payload")[0].as_py())
    assert payload[0]["modifier"] == "cup"


def test_the_raw_payload_is_preserved_unmodified():
    """Bronze is source-preserving: cleaning happens in Silver, so anything USDA sends must
    survive the write intact and stay re-derivable."""
    portion = {"measureUnit": {"name": "undetermined"}, "modifier": "cup, chopped",
               "gramWeight": 160.0, "amount": 1.0, "sequenceNumber": 3, "id": 81234}

    table = build_bronze_table([("10", [portion])])

    assert json.loads(table.column("raw_payload")[0].as_py()) == [portion]


def test_a_read_timeout_is_retried_not_fatal(monkeypatch):
    """This is the bug that killed the first real pull: `_post_batch` retried HTTP statuses but
    not transport exceptions, so a single ReadTimeout 32 minutes into a ~410-request run raised
    straight through and lost everything. Over a pull that long, meeting one is near-certain."""
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ReadTimeout("read timed out")
        return _Response(200, [{"fdcId": 1, "foodPortions": []}])

    monkeypatch.setattr(requests, "post", flaky)
    slept = []

    assert _post_batch("key", ["1"], slept.append) == [{"fdcId": 1, "foodPortions": []}]
    assert len(calls) == 2 and slept == [1]


def test_a_dropped_connection_is_retried(monkeypatch):
    """Same class of failure, different exception — both subclass requests.RequestException."""
    responses = [requests.ConnectionError("reset by peer"),
                 _Response(200, [])]

    def flaky(*args, **kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(requests, "post", flaky)

    assert _post_batch("key", ["1"], lambda _: None) == []


def test_persistent_transport_failure_still_gives_up(monkeypatch):
    """Retrying forever would hang the pipeline; failing loudly beats a partial table that
    looks complete."""
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ReadTimeout("nope")))

    with pytest.raises(RuntimeError, match="after 5 retries"):
        _post_batch("key", ["1"], lambda _: None)


def test_a_resumed_run_only_fetches_what_is_missing(tmp_path, monkeypatch):
    """~410 requests take ~40 minutes and the write happens only after the last one, so the
    first attempt lost everything to a single timeout at minute 32. Caching each batch as it
    lands turns that into one lost batch."""
    cache = tmp_path / "cache.jsonl"
    asked = []

    def fake_post(key, ids, sleep):
        asked.extend(ids)
        return [{"fdcId": int(i), "foodPortions": [_portion("undetermined", "cup", 240.0)]}
                for i in ids]

    monkeypatch.setattr("pantryiq.ingestion.portions._post_batch", fake_post)

    first = dict(fetch_portions("key", ["1", "2", "3"], batch_size=2, cache_path=cache))
    assert asked == ["1", "2", "3"]

    asked.clear()
    second = dict(fetch_portions("key", ["1", "2", "3", "4"], batch_size=2, cache_path=cache))

    assert asked == ["4"]          # 1-3 came from the cache
    assert second["1"] == first["1"]
    assert len(second) == 4


def test_the_cache_survives_a_truncated_final_line(tmp_path, monkeypatch):
    """A process killed mid-write leaves a partial JSON line. Raising on it would make the cache
    a liability — the whole point is that an interrupted run recovers."""
    cache = tmp_path / "cache.jsonl"
    cache.write_text('{"fdc_id": "1", "portions": []}\n{"fdc_id": "2", "por')

    monkeypatch.setattr("pantryiq.ingestion.portions._post_batch",
                        lambda key, ids, sleep: [{"fdcId": int(i), "foodPortions": []}
                                                 for i in ids])
    assert load_cache(cache) == {"1": []}

    result = dict(fetch_portions("key", ["1", "2"], cache_path=cache))
    assert set(result) == {"1", "2"}      # 2 was simply re-fetched


def test_caching_can_be_switched_off(tmp_path, monkeypatch):
    """The tests that assert on request batching must not pick up a cache from a previous run."""
    monkeypatch.setattr("pantryiq.ingestion.portions._post_batch",
                        lambda key, ids, sleep: [{"fdcId": int(i), "foodPortions": []}
                                                 for i in ids])

    list(fetch_portions("key", ["1"], cache_path=None))

    assert not (tmp_path / "cache.jsonl").exists()
