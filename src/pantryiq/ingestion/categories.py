"""Ingest USDA `foodCategory` — the authoritative food group, so dietary tags can stop guessing.

`gold.recipe_tags` stated `vegetarian` / `vegan` / `gluten-free` as the **absence of a
disqualifying substring in `display_name`**. A denylist cannot support a safety claim: every
product name nobody thought of is a silent false positive. Two were found by joining Phase 4's
new recipe titles against the tags — an independent signal that did not exist when the predicate
was written:

    recipenlg:9231  "Meat Loaf"  tagged vegetarian   `hamburger` -> "BURGER KING, Hamburger"
    recipenlg:10493 "Meat Loaf"  tagged gluten-free  `quaker oat` -> "Cereals, QUAKER, QUAKER
                                                                     MultiGrain Oatmeal, dry"

Neither string contains a listed meat or gluten token (`'%ham,%'` is comma-anchored so it will
not match `graham`; the gluten list has `oats`, not `oatmeal`). Measured against titles, **13 of
604 vegetarian and 9 of 147 vegan recipes name meat in their own title** — and that is a lower
bound, since it only catches recipes whose title gives them away.

`foodCategory` replaces the denylist with an **allowlist**: a recipe is tagged only when every
one of its entities sits in a category known to qualify. Both failures above are then fixed by
construction rather than by adding two more patterns — "Fast Foods" and "Breakfast Cereals" are
simply not on the vegetarian or gluten-free lists, so those recipes decline instead of lying.

**Neither existing Bronze table carries it.** `raw_usda_foods` came from `/foods/list`, whose
records stop at `foodNutrients`; `raw_usda_portions` cached `format=full` responses but kept only
`fdcId` and `foodPortions`, discarding the rest. So the category costs its own pull — the same
~410 requests over the same frozen 8,187 ids, reusing this package's batching and backoff.

**This writes a THIRD table and leaves the other two untouched**, for the reason `portions.py`
gives: re-pulling `raw_usda_foods` would move the frozen entity set that the 300 gold labels and
every §7-§12 metric are pinned to. Categories join on `fdc_id`.

Idempotent, resumable, and a 0-row result is refused rather than atomically wiping the table.

Run directly:  uv run python -m pantryiq.ingestion.categories
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.table import Table

from pantryiq.ingestion.portions import BATCH_SIZE, _post_batch, food_ids
from pantryiq.ingestion.usda import NAMESPACE, SOURCE, _require_api_key

TABLE = f"{NAMESPACE}.raw_usda_categories"
# Same reason as the portions cache: ~410 sequential requests, and without a cache one timeout
# at minute 32 costs the entire run.
CACHE_PATH = Path("data/usda_categories_cache.jsonl")

SCHEMA = pa.schema(
    [
        ("fdc_id", pa.string()),
        ("source", pa.string()),
        ("ingested_at", pa.timestamp("us", tz="UTC")),
        ("food_category", pa.string()),  # "" when USDA publishes none — see extract_category
    ]
)


def extract_category(food: dict) -> str:
    """The food's category description, across the three shapes FDC actually returns.

    SR Legacy and Foundation foods carry `foodCategory` as an object with a `description`;
    some records flatten it to a bare string; Branded foods use `brandedFoodCategory` instead.
    An absent category yields `""` rather than a guess — under an allowlist, unknown correctly
    means "do not tag", so a missing value is safe by default.
    """
    category = food.get("foodCategory")
    if isinstance(category, dict):
        return str(category.get("description") or "").strip()
    if isinstance(category, str) and category.strip():
        return category.strip()
    return str(food.get("brandedFoodCategory") or "").strip()


def load_cache(path: Path | str = CACHE_PATH) -> dict[str, str]:
    """Everything already fetched, so a re-run resumes instead of starting over.

    A truncated final line (the process died mid-write) is dropped rather than raising — the id
    it belonged to is simply re-fetched.
    """
    path = Path(path)
    if not path.exists():
        return {}
    cached: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cached[record["fdc_id"]] = record["food_category"]
    return cached


def fetch_categories(api_key: str, fdc_ids: Iterable[str], batch_size: int = BATCH_SIZE,
                     sleep=time.sleep, cache_path: Path | str | None = CACHE_PATH
                     ) -> Iterator[tuple[str, str]]:
    """Yield (fdc_id, food_category) for every id, batching the requests and caching as it goes.

    Foods USDA publishes no category for yield `""` rather than being skipped: "we asked and
    there is none" is a different fact from "we never asked", and only the first is checkable.
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
            returned = {str(food.get("fdcId", "")): extract_category(food)
                        for food in _post_batch(api_key, batch, sleep)}
            for fdc_id in batch:
                cached[fdc_id] = returned.get(fdc_id, "")
                if handle:
                    handle.write(json.dumps(
                        {"fdc_id": fdc_id, "food_category": cached[fdc_id]},
                        ensure_ascii=False) + "\n")
            if handle:
                handle.flush()  # survive a kill -9, not just a clean exception
    finally:
        if handle:
            handle.close()

    for fdc_id in ids:
        yield (fdc_id, cached.get(fdc_id, ""))


def build_bronze_table(records: Iterable[tuple[str, str]]) -> pa.Table:
    """Shape (fdc_id, category) pairs into the Bronze schema."""
    ingested_at = datetime.now(timezone.utc)
    fdc_ids: list[str] = []
    categories: list[str] = []
    for fdc_id, category in records:
        fdc_ids.append(str(fdc_id))
        categories.append(category)
    return pa.table(
        {
            "fdc_id": fdc_ids,
            "source": [SOURCE] * len(fdc_ids),
            "ingested_at": [ingested_at] * len(fdc_ids),
            "food_category": categories,
        },
        schema=SCHEMA,
    )


def ingest_usda_categories(catalog: Catalog, records: Iterable[tuple[str, str]] | None = None,
                           api_key: str | None = None) -> Table:
    """Land categories into `bronze.raw_usda_categories` (idempotent overwrite); return the table.

    Pass `records` to skip the network (used by tests); otherwise fetch from the API.
    """
    if records is None:
        records = fetch_categories(api_key or _require_api_key(), food_ids(catalog))
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
    from collections import Counter

    from pantryiq.lakehouse.catalog import get_catalog

    catalog = get_catalog()
    table = ingest_usda_categories(catalog)
    rows = table.scan().to_arrow()
    categories = rows.column("food_category").to_pylist()
    known = [value for value in categories if value]

    print(f"bronze.raw_usda_categories: {len(categories):,} rows landed")
    print(f"  with a category : {len(known):,} ({100 * len(known) / len(categories):.1f}%)")
    print(f"  distinct values : {len(set(known)):,}")
    print("\n  the 25 most common — the vocabulary the tag allowlists are built from:")
    for value, count in Counter(known).most_common(25):
        print(f"    {count:6,}  {value}")


if __name__ == "__main__":
    main()
