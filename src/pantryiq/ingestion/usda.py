"""Ingest USDA FoodData Central foods into the Bronze Iceberg layer.

Canonical set = Foundation + SR Legacy (~8K foods); Branded is intentionally excluded.
Pulls the paginated /foods/list endpoint (abridged records already carry the macros we
need), preserving each food unmodified as a JSON `raw_payload` plus lineage. Idempotent:
a re-run overwrites the table. The API key is read from `.env` (USDA_API_KEY).

**Single-source assumption:** the unfiltered `overwrite` replaces the WHOLE table, which is
correct only because `usda_fdc` is the sole source of `bronze.raw_usda_foods`. If a second
nutrition source is ever added, switch to `overwrite(data, EqualTo("source", SOURCE))`.

Run directly to land the canonical set:  uv run python -m pantryiq.ingestion.usda
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone

import pyarrow as pa
import requests
from dotenv import load_dotenv
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.table import Table

SOURCE = "usda_fdc"
NAMESPACE = "bronze"
TABLE = f"{NAMESPACE}.raw_usda_foods"
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DATA_TYPES = ("Foundation", "SR Legacy")  # canonical set; Branded excluded by design
PAGE_SIZE = 200
MAX_RETRIES = 5
RETRYABLE = {429, 500, 502, 503, 504}

SCHEMA = pa.schema(
    [
        ("fdc_id", pa.string()),
        ("source", pa.string()),
        ("data_type", pa.string()),
        ("description", pa.string()),
        ("ingested_at", pa.timestamp("us", tz="UTC")),
        ("raw_payload", pa.string()),
    ]
)


def _require_api_key() -> str:
    load_dotenv()
    key = os.environ.get("USDA_API_KEY")
    if not key:
        raise RuntimeError("USDA_API_KEY not set — put it in .env")
    return key


def _get_page(api_key: str, data_type: str, page: int, page_size: int, sleep) -> list[dict]:
    """One /foods/list page, with exponential backoff on rate-limit / 5xx."""
    params = {
        "dataType": data_type,
        "pageSize": page_size,
        "pageNumber": page,
        "api_key": api_key,
    }
    for attempt in range(MAX_RETRIES):
        resp = requests.get(f"{BASE_URL}/foods/list", params=params, timeout=30)
        if resp.status_code == 200:
            body = resp.json()
            if not isinstance(body, list):
                raise RuntimeError(f"unexpected /foods/list payload (not a list): {body!r:.200}")
            return body
        if resp.status_code in RETRYABLE:
            retry_after = resp.headers.get("Retry-After", "")
            # Cap Retry-After: the API has served absurd values, and an uncapped sleep stalls
            # the whole pull (999999s ~= 11 days).
            wait = min(int(retry_after), 60) if retry_after.isdigit() else min(2**attempt, 30)
            sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"FDC /foods/list failed after {MAX_RETRIES} retries "
                       f"(dataType={data_type}, page={page})")


def fetch_foods(
    api_key: str,
    data_types: Iterable[str] = DATA_TYPES,
    page_size: int = PAGE_SIZE,
    max_pages: int | None = None,
    sleep=time.sleep,
) -> Iterator[dict]:
    """Yield every food across `data_types`, paginating until a short/empty page."""
    for data_type in data_types:
        page = 1
        while max_pages is None or page <= max_pages:
            batch = _get_page(api_key, data_type, page, page_size, sleep)
            yield from batch
            if len(batch) < page_size:
                break
            page += 1


def build_bronze_table(foods: Iterable[dict]) -> pa.Table:
    """Shape an iterable of FDC food dicts into the Bronze schema."""
    ingested_at = datetime.now(timezone.utc)
    fdc_ids: list[str] = []
    data_types: list[str] = []
    descriptions: list[str] = []
    payloads: list[str] = []
    for food in foods:
        fdc_ids.append(str(food.get("fdcId", "")))
        data_types.append(food.get("dataType") or "")
        descriptions.append(food.get("description") or "")
        payloads.append(json.dumps(food, ensure_ascii=False))
    n = len(fdc_ids)
    return pa.table(
        {
            "fdc_id": fdc_ids,
            "source": [SOURCE] * n,
            "data_type": data_types,
            "description": descriptions,
            "ingested_at": [ingested_at] * n,
            "raw_payload": payloads,
        },
        schema=SCHEMA,
    )


def ingest_usda_foods(
    catalog: Catalog,
    foods: Iterable[dict] | None = None,
    api_key: str | None = None,
    page_size: int = PAGE_SIZE,
    max_pages: int | None = None,
) -> Table:
    """Land USDA foods into `bronze.raw_usda_foods` (idempotent overwrite); return the table.

    Pass `foods` to skip the network (used by tests); otherwise fetch from the API.
    """
    if foods is None:
        foods = fetch_foods(api_key or _require_api_key(), page_size=page_size, max_pages=max_pages)
    data = build_bronze_table(foods)
    if data.num_rows == 0:
        raise ValueError(
            f"refusing to write {TABLE} with 0 rows — an empty overwrite would atomically "
            "wipe the table (check the API key, dataType, and rate-limit responses)"
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
    table = ingest_usda_foods(catalog)
    print(f"bronze.raw_usda_foods: {table.scan().to_arrow().num_rows} rows landed")


if __name__ == "__main__":
    main()
