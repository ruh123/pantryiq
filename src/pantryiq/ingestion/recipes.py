"""Ingest RecipeNLG recipes into the Bronze Iceberg layer (raw, source-preserving).

Bronze preserves each source row unmodified as a JSON `raw_payload` plus lineage
(recipe_id, source, source_url, ingested_at). Cleaning/typing happens later in Silver.
Idempotent: a re-run overwrites the table rather than appending, so row counts stay stable.

**Single-source assumption:** the unfiltered `overwrite` replaces the WHOLE table, which is
correct only because `recipenlg` is the sole source of `bronze.raw_recipes`. If a second
recipe source is ever added, switch to `overwrite(data, EqualTo("source", SOURCE))` or the
new source will delete this one.

Run directly to land the subset:  uv run python -m pantryiq.ingestion.recipes
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.table import Table

SOURCE = "recipenlg"
NAMESPACE = "bronze"
TABLE = f"{NAMESPACE}.raw_recipes"
DEFAULT_CSV = Path("data/raw/recipenlg/full_dataset.csv")
DEFAULT_LIMIT = 15_000  # subset-first; None = full corpus

csv.field_size_limit(10_000_000)

SCHEMA = pa.schema(
    [
        ("recipe_id", pa.string()),
        ("source", pa.string()),
        ("source_url", pa.string()),
        ("ingested_at", pa.timestamp("us", tz="UTC")),
        ("raw_payload", pa.string()),
    ]
)


def build_bronze_table(
    csv_path: Path | str = DEFAULT_CSV,
    limit: int | None = DEFAULT_LIMIT,
) -> pa.Table:
    """Read up to `limit` RecipeNLG rows and shape them into the Bronze schema.

    The RecipeNLG CSV's leading (unnamed) column is a row index; we use it as the
    within-source id and keep the entire row as `raw_payload`.
    """
    ingested_at = datetime.now(timezone.utc)
    recipe_ids: list[str] = []
    urls: list[str] = []
    payloads: list[str] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            row_id = (row.get("") or str(i)).strip()
            recipe_ids.append(f"{SOURCE}:{row_id}")
            urls.append(row.get("link") or "")
            payloads.append(json.dumps(row, ensure_ascii=False))
    n = len(recipe_ids)
    return pa.table(
        {
            "recipe_id": recipe_ids,
            "source": [SOURCE] * n,
            "source_url": urls,
            "ingested_at": [ingested_at] * n,
            "raw_payload": payloads,
        },
        schema=SCHEMA,
    )


def ingest_recipes(
    catalog: Catalog,
    csv_path: Path | str = DEFAULT_CSV,
    limit: int | None = DEFAULT_LIMIT,
) -> Table:
    """Land RecipeNLG into `bronze.raw_recipes` via idempotent overwrite; return the table."""
    try:
        catalog.create_namespace(NAMESPACE)
    except NamespaceAlreadyExistsError:
        pass
    data = build_bronze_table(csv_path, limit)
    if data.num_rows == 0:
        raise ValueError(
            f"refusing to write {TABLE} with 0 rows — an empty overwrite would atomically "
            f"wipe the table (check that {csv_path} is the RecipeNLG export)"
        )
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
    table = ingest_recipes(catalog)
    print(f"bronze.raw_recipes: {table.scan().to_arrow().num_rows} rows landed")


if __name__ == "__main__":
    main()
