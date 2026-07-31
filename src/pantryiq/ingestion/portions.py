"""Ingest USDA `foodPortions` — the gram weights that make recipe quantities computable.

Every nutrient in `silver.usda_foods` is **per 100g**, and recipe lines say "2 cups" or "1 (16
oz.) can". Without a gram weight per (food, measure) there is no per-recipe nutrition, no
`data_trust_score`, and nothing for the Phase-4 agent to answer from — so this is the first
thing Phase 3 needs, ahead of any dbt work.

**The abridged payload does not carry it.** `bronze.raw_usda_foods` came from `/foods/list`,
whose records stop at `foodNutrients`. But the *bulk download is not required*: `POST /v1/foods`
with `format=full` returns `foodPortions`, and it accepts 20 ids per call — all 8,187 foods in
~410 requests, well inside the ~1,000/hr key limit. Measured coverage on the 200 most-hit
entities (87% of resolved occurrences): 97.1% carry at least one portion.

**This writes a NEW table and leaves `bronze.raw_usda_foods` untouched.** Re-pulling those 8,187
foods in full format would change the frozen entity set that the 300 gold labels and every
§7-§12 metric are pinned to. The portions join on `fdc_id` instead.

Idempotent: a re-run overwrites the table, and a 0-row result is refused rather than atomically
wiping it. The API key is read from `.env` (USDA_API_KEY).

Run directly:  uv run python -m pantryiq.ingestion.portions
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import requests
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.table import Table

from pantryiq.ingestion.usda import (
    BASE_URL,
    MAX_RETRIES,
    NAMESPACE,
    RETRYABLE,
    SOURCE,
    _require_api_key,
)
from pantryiq.ingestion.usda import TABLE as FOODS_TABLE

TABLE = f"{NAMESPACE}.raw_usda_portions"
# The /v1/foods POST endpoint caps at 20 ids per request.
BATCH_SIZE = 20
# Batches land here as they arrive. ~410 requests take ~40 minutes, and the first attempt lost
# all of it to one timeout at minute 32 — the write only happens after every batch is in, so
# any failure costs the whole run. With the cache a failure costs one batch, a re-run is
# near-instant, and the pull becomes reproducible for anyone cloning the repo.
CACHE_PATH = Path("data/usda_portions_cache.jsonl")
# Full-format responses carry every nutrient for 20 foods — roughly 1 MB per batch, so the
# read is slow enough that the foods pull's 30s would clip healthy responses.
TIMEOUT = 120

SCHEMA = pa.schema(
    [
        ("fdc_id", pa.string()),
        ("source", pa.string()),
        ("ingested_at", pa.timestamp("us", tz="UTC")),
        ("raw_payload", pa.string()),  # the foodPortions array, unmodified
    ]
)


def _post_batch(api_key: str, fdc_ids: list[str], sleep) -> list[dict]:
    """One /foods POST for up to 20 ids, with exponential backoff on rate-limit / 5xx.

    The retry shape mirrors `usda._get_page` rather than sharing it: that one is a paginated GET
    with query params, this is a POST with a JSON body, and folding both into one helper would
    cost more in indirection than the ten duplicated lines are worth.
    """
    body = {"fdcIds": fdc_ids, "format": "full"}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(f"{BASE_URL}/foods", params={"api_key": api_key},
                                 json=body, timeout=TIMEOUT)
        except requests.RequestException:
            # Transport-level failures must retry, not just HTTP statuses. This pull is ~410
            # sequential requests over ~30 minutes; meeting one read timeout or dropped
            # connection is near-certain, and an unhandled one loses the entire run. (It did:
            # the first attempt died at 32 minutes on a ReadTimeout with nothing written.)
            sleep(min(2**attempt, 30))
            continue
        if resp.status_code == 200:
            payload = resp.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"unexpected /foods payload (not a list): {payload!r:.200}")
            return payload
        if resp.status_code in RETRYABLE:
            retry_after = resp.headers.get("Retry-After", "")
            # Capped for the same reason as the foods pull: an uncapped Retry-After stalls
            # the whole run (999999s ~= 11 days).
            wait = min(int(retry_after), 60) if retry_after.isdigit() else min(2**attempt, 30)
            sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"FDC /foods failed after {MAX_RETRIES} retries "
                       f"(first id={fdc_ids[0] if fdc_ids else 'none'})")


def food_ids(catalog: Catalog) -> list[str]:
    """The frozen canonical entity set, read back from Bronze — sorted for a stable pull order."""
    foods = catalog.load_table(FOODS_TABLE).scan().to_arrow()
    return sorted(set(foods.column("fdc_id").to_pylist()))


def load_cache(path: Path | str = CACHE_PATH) -> dict[str, list]:
    """Everything already fetched, so a re-run resumes instead of starting over.

    A truncated final line (the process died mid-write) is dropped rather than raising — the
    id it belonged to is simply re-fetched.
    """
    path = Path(path)
    if not path.exists():
        return {}
    cached: dict[str, list] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cached[record["fdc_id"]] = record["portions"]
    return cached


def fetch_portions(api_key: str, fdc_ids: Iterable[str], batch_size: int = BATCH_SIZE,
                   sleep=time.sleep, cache_path: Path | str | None = CACHE_PATH
                   ) -> Iterator[tuple[str, list]]:
    """Yield (fdc_id, foodPortions) for every id, batching the requests and caching as it goes.

    Foods with no portions yield an empty list rather than being skipped: "we asked and USDA has
    none" is a different fact from "we never asked", and only the first can be verified later.

    Each batch is appended to `cache_path` and flushed before the next request, so an
    interrupted run resumes from the last completed batch. Pass `cache_path=None` to disable.
    """
    ids = list(fdc_ids)
    cached = load_cache(cache_path) if cache_path else {}
    outstanding = [fdc_id for fdc_id in ids if fdc_id not in cached]

    handle = None
    if cache_path and outstanding:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        handle = cache_path.open("a", encoding="utf-8")
    try:
        for start in range(0, len(outstanding), batch_size):
            batch = outstanding[start:start + batch_size]
            returned = {str(food.get("fdcId", "")): food.get("foodPortions") or []
                        for food in _post_batch(api_key, batch, sleep)}
            for fdc_id in batch:
                cached[fdc_id] = returned.get(fdc_id, [])
                if handle:
                    handle.write(json.dumps(
                        {"fdc_id": fdc_id, "portions": cached[fdc_id]}, ensure_ascii=False) + "\n")
            if handle:
                handle.flush()  # survive a kill -9, not just a clean exception
    finally:
        if handle:
            handle.close()

    for fdc_id in ids:
        yield (fdc_id, cached.get(fdc_id, []))


def build_bronze_table(records: Iterable[tuple[str, list]]) -> pa.Table:
    """Shape (fdc_id, portions) pairs into the Bronze schema."""
    ingested_at = datetime.now(timezone.utc)
    fdc_ids: list[str] = []
    payloads: list[str] = []
    for fdc_id, portions in records:
        fdc_ids.append(str(fdc_id))
        payloads.append(json.dumps(portions, ensure_ascii=False))
    return pa.table(
        {
            "fdc_id": fdc_ids,
            "source": [SOURCE] * len(fdc_ids),
            "ingested_at": [ingested_at] * len(fdc_ids),
            "raw_payload": payloads,
        },
        schema=SCHEMA,
    )


def ingest_usda_portions(catalog: Catalog, records: Iterable[tuple[str, list]] | None = None,
                         api_key: str | None = None) -> Table:
    """Land portions into `bronze.raw_usda_portions` (idempotent overwrite); return the table.

    Pass `records` to skip the network (used by tests); otherwise fetch from the API.
    """
    if records is None:
        records = fetch_portions(api_key or _require_api_key(), food_ids(catalog))
    data = build_bronze_table(records)
    if data.num_rows == 0:
        raise ValueError(
            f"refusing to write {TABLE} with 0 rows — an empty overwrite would atomically "
            "wipe the table (check the API key and rate-limit responses)"
        )
    try:
        catalog.create_namespace(NAMESPACE)
    except NamespaceAlreadyExistsError:
        pass
    try:
        table = catalog.load_table(TABLE)
        table.overwrite(data)
    except NoSuchTableError:
        table = catalog.create_table(TABLE, schema=data.schema)
        table.append(data)
    return catalog.load_table(TABLE)


def main() -> None:
    from pantryiq.lakehouse.catalog import get_catalog

    catalog = get_catalog()
    table = ingest_usda_portions(catalog)
    rows = table.scan().to_arrow()
    payloads = [json.loads(value) for value in rows.column("raw_payload").to_pylist()]
    with_portions = sum(1 for portions in payloads if portions)
    total = len(payloads)

    print(f"bronze.raw_usda_portions: {total:,} rows landed")
    print(f"  with >=1 portion : {with_portions:,} ({100 * with_portions / total:.1f}%)")
    print(f"  portions in total: {sum(len(portions) for portions in payloads):,}")


if __name__ == "__main__":
    main()
