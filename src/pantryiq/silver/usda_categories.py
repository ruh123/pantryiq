"""USDA food categories into Silver — the authoritative field the dietary tags now rest on.

`bronze.raw_usda_categories` -> `silver.usda_food_categories`. Thin by design: the Bronze row is
already one category per food, so there is nothing to parse, and the value of this module is that
Gold reads Silver rather than reaching into the lakehouse.

**Why this exists at all.** `gold.recipe_tags` stated `vegetarian` / `vegan` / `gluten-free` as
the absence of a disqualifying substring in `display_name`. A denylist cannot support a safety
claim — every product name nobody anticipated is a silent false positive, and two were found by
joining Phase 4's recipe titles against the tags: a "Meat Loaf" tagged vegetarian because
`hamburger` resolved to "BURGER KING, Hamburger" (no listed meat word), and another tagged
gluten-free because `quaker oat` resolved to "Cereals, QUAKER, QUAKER MultiGrain Oatmeal" (the
list had `oats`, not `oatmeal`). Measured against titles, 13 of 604 vegetarian and 9 of 147 vegan
recipes name meat in their own title, and that is a lower bound.

**The vocabulary is closed, which is the whole point.** USDA publishes exactly 25 food groups
across these 8,187 foods, 99.0% of which carry one. A closed vocabulary can be reasoned about
once and audited; an open set of product names cannot. `models/gold/recipe_tags.sql` allows
specific groups rather than denying specific words, so an unrecognised food declines the tag
instead of silently qualifying for it.

Run directly:  uv run python -m pantryiq.silver.usda_categories
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog

DEFAULT_DB = Path("data/pantryiq.duckdb")
BRONZE_TABLE = "bronze.raw_usda_categories"


def build_rows(catalog: Catalog) -> pa.Table:
    """One row per food: fdc_id, food_category. An absent category stays NULL, never ''."""
    from pantryiq.lakehouse.catalog import scan_with_duckdb

    bronze = scan_with_duckdb(catalog.load_table(BRONZE_TABLE))
    fdc_ids = [str(value) for value in bronze.column("fdc_id").to_pylist()]
    # NULL rather than "": under an allowlist, unknown means "do not tag", and a NULL says that
    # louder than an empty string that could be mistaken for a real category downstream.
    categories = [value.strip() or None for value in bronze.column("food_category").to_pylist()]
    return pa.table({"fdc_id": fdc_ids, "food_category": categories})


def write_silver(rows: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    """Write `silver.usda_food_categories`, replacing it (idempotent re-run)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("category_rows", rows)
        con.execute("CREATE OR REPLACE TABLE silver.usda_food_categories AS "
                    "SELECT * FROM category_rows")
    finally:
        con.close()
    return db_path


def main() -> None:
    from pantryiq.lakehouse.catalog import get_catalog
    from pantryiq.lakehouse.writelock import warehouse_lock

    with warehouse_lock():
        write_silver(build_rows(get_catalog()))
        con = duckdb.connect(str(DEFAULT_DB), read_only=True)
        try:
            total, known = con.execute(
                "SELECT count(*), count(food_category) FROM silver.usda_food_categories"
            ).fetchone()
            orphans = con.execute(
                """
                SELECT count(*) FROM silver.usda_foods f
                LEFT JOIN silver.usda_food_categories c USING (fdc_id)
                WHERE c.fdc_id IS NULL
                """
            ).fetchone()[0]
            groups = con.execute(
                """
                SELECT food_category, count(*) FROM silver.usda_food_categories
                WHERE food_category IS NOT NULL GROUP BY 1 ORDER BY 2 DESC
                """
            ).fetchall()
        finally:
            con.close()

    print(f"silver.usda_food_categories: {total:,} foods")
    print(f"  with a category : {known:,} ({100 * known / total:.1f}%)")
    print(f"  distinct groups : {len(groups)}")
    print(f"  foods in usda_foods with no category row: {orphans}  (must be 0)")


if __name__ == "__main__":
    main()
