"""Recipe titles and source URLs — the one thing Gold could not say about a recipe.

`bronze.raw_recipes` -> `silver.recipe_meta`. Every Gold table keys on `recipe_id`, and none of
them carries a title, so the Phase-4 agent could retrieve a perfect match and then be unable to
name it. The title is in the Bronze payload for all 15,000 recipes; this lifts it into the
serving path.

**Read Bronze through `scan_with_duckdb`, never a parquet glob.** The `raw_recipes` data
directory holds TWO snapshots of 15,000 rows each — an orphan from a DELETE+APPEND sequence —
with identical `recipe_id`s. `read_parquet('.../data/*.parquet')` returns 30,000 rows and
silently doubles every downstream join; `scan_with_duckdb` resolves the table's *current*
metadata and returns 15,000. Verified both ways before this module was written.

Two small cleanings, because Silver is where cleaning belongs:

- **Titles are stripped.** The corpus ships them with trailing whitespace (`'Reeses
  Cups(Candy)  '`), which would show up verbatim in an agent's answer.
- **Source URLs are scheme-less** in the payload (`www.cookbooks.com/...`). They are left exactly
  as the source wrote them — Bronze is source-preserving and inventing an `https://` would be
  asserting a fact about a site nobody checked.

**Titles are untrusted third-party text.** They come from a public scrape and flow into a Claude
prompt in Phase 4, so every consumer must treat them as data — see `agent/generate.py`, which
wraps them in a delimited tag rather than interpolating them into instructions.

Run directly:  uv run python -m pantryiq.silver.recipe_meta
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog

DEFAULT_DB = Path("data/pantryiq.duckdb")
BRONZE_TABLE = "bronze.raw_recipes"


def build_rows(catalog: Catalog) -> pa.Table:
    """One row per recipe: id, title, source URL, directions.

    `directions` exists for `gold.recipe_tags`, not for display. The corpus ships some recipes
    with a truncated ingredient list — "Pickled Bologna" lists vinegar, sugar, salt and pickling
    spice and no bologna — so a recipe can be fully weighed, pass the coverage gate, and still be
    missing the ingredient that disqualifies it. The directions name it ("remove skin from 2
    rings bologna"), which makes them the only place in the corpus that gap is visible.
    """
    from pantryiq.lakehouse.catalog import scan_with_duckdb

    bronze = scan_with_duckdb(catalog.load_table(BRONZE_TABLE))
    recipe_ids: list[str] = []
    titles: list[str | None] = []
    urls: list[str | None] = []
    directions: list[str | None] = []

    for recipe_id, payload in zip(bronze.column("recipe_id").to_pylist(),
                                  bronze.column("raw_payload").to_pylist()):
        record = json.loads(payload)
        title = (record.get("title") or "").strip()
        url = (record.get("link") or "").strip()
        # The payload stores this as a JSON *string*, not a list. Joining it as if it were a
        # list spaces out every character ("b o i l   i n g r e d i e n t s") and silently
        # destroys the field — a control run caught exactly that, matching 0 of 15,000 recipes.
        steps = record.get("directions") or ""
        text = steps if isinstance(steps, str) else " ".join(steps)
        recipe_ids.append(recipe_id)
        titles.append(title or None)
        urls.append(url or None)
        directions.append(text.strip().lower() or None)

    return pa.table({"recipe_id": recipe_ids, "title": titles, "source_url": urls,
                     "directions": directions})


def write_silver(rows: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    """Write `silver.recipe_meta`, replacing it (idempotent re-run)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("meta_rows", rows)
        con.execute("CREATE OR REPLACE TABLE silver.recipe_meta AS SELECT * FROM meta_rows")
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
            total, titled, linked = con.execute(
                "SELECT count(*), count(title), count(source_url) FROM silver.recipe_meta"
            ).fetchone()
            orphans = con.execute(
                """
                SELECT count(*) FROM silver.recipe_ingredient_lines l
                LEFT JOIN silver.recipe_meta m USING (recipe_id)
                WHERE m.recipe_id IS NULL
                """
            ).fetchone()[0]
            sample = con.execute(
                "SELECT recipe_id, title FROM silver.recipe_meta ORDER BY recipe_id LIMIT 3"
            ).fetchall()
        finally:
            con.close()

    print(f"silver.recipe_meta: {total:,} recipes")
    print(f"  with a title      : {titled:,} ({100 * titled / total:.1f}%)")
    print(f"  with a source URL : {linked:,} ({100 * linked / total:.1f}%)")
    print(f"  ingredient lines with no meta row: {orphans}  (must be 0)")
    for recipe_id, title in sample:
        print(f"    {recipe_id:16} {title!r}")


if __name__ == "__main__":
    main()
