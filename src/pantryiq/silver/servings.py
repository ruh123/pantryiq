"""Extract a servings count from recipe text, for the minority of recipes that state one.

RecipeNLG has no servings column (Master Prompt §9 risk 1), so per-serving nutrition has no
denominator unless the recipe says one in its own directions — "Serves 6", "Makes 8 servings",
"Yields 4 dozen". Measured on the 15K subset: 18.3% of recipes mention a yield word at all and
**15.4% state an extractable integer**.

**Nothing is estimated.** A servings figure could be guessed from total mass (say ~400 g per
serving), which would populate the column for every recipe — and put an invented denominator
inside every per-serving number the agent ever quotes. The rest stay NULL, and `kcal_per_serving`
is NULL with them, so a missing denominator is visible instead of fabricated.

**"Makes 4 dozen cookies" is a yield, not a serving count**, and reading it as 4 would overstate
per-serving energy twelvefold. Dozen-qualified yields are multiplied out; yields in units that
are not portions (loaves, pans, quarts) are rejected rather than guessed at.

Run:  uv run python -m pantryiq.silver.servings
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog

DEFAULT_DB = Path("data/pantryiq.duckdb")
BRONZE_TABLE = "bronze.raw_recipes"

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}
_COUNT = r"(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"

# "Serves 6", "Makes about 8 servings", "Yield: 4 dozen cookies".
_YIELD = re.compile(
    rf"\b(?:serves?|serving[s]?\s*:|makes|yield[s]?)\s*:?\s*"
    rf"(?:about|approximately|around)?\s*{_COUNT}\s*(?:to|-|–)?\s*(?:\d{{1,3}})?\s*"
    rf"(dozen)?\s*([a-z]+)?",
    re.I)

# A yield counted in these is not a serving count — "makes 2 loaves" says nothing about how many
# people it feeds, and treating it as 2 servings would overstate per-serving energy hugely.
NON_PORTION_UNITS = {
    "loaf", "loaves", "pan", "pans", "cake", "cakes", "pie", "pies", "crust", "crusts",
    "quart", "quarts", "pint", "pints", "gallon", "gallons", "cup", "cups", "jar", "jars",
    "pound", "pounds", "lb", "lbs", "batch", "batches", "recipe", "recipes",
}
# An implausible count is more likely a misparse than a real yield.
MAX_SERVINGS = 200


def parse_servings(text: str) -> int | None:
    """The stated servings count, or None when the text does not give one usably."""
    match = _YIELD.search(text or "")
    if not match:
        return None
    raw, dozen, unit = match.group(1), match.group(2), (match.group(3) or "").lower()
    count = _NUMBER_WORDS.get(raw.lower()) if not raw.isdigit() else int(raw)
    if not count:
        return None
    if dozen:
        count *= 12
        unit = ""  # "4 dozen cookies" — the cookies are the portions
    if unit in NON_PORTION_UNITS:
        return None
    return count if 0 < count <= MAX_SERVINGS else None


def build_rows(catalog: Catalog) -> pa.Table:
    """One row per recipe, with `servings` where the text states one."""
    bronze = catalog.load_table(BRONZE_TABLE).scan().to_arrow()
    recipe_ids, servings = [], []
    for recipe_id, payload in zip(bronze.column("recipe_id").to_pylist(),
                                  bronze.column("raw_payload").to_pylist()):
        record = json.loads(payload)
        text = f"{record.get('title', '')} {record.get('directions', '')}"
        recipe_ids.append(recipe_id)
        servings.append(parse_servings(text))
    return pa.table({"recipe_id": recipe_ids, "servings": servings})


def write_silver(rows: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    """Write `silver.recipe_servings`, replacing it (idempotent re-run)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("serving_rows", rows)
        con.execute("CREATE OR REPLACE TABLE silver.recipe_servings AS "
                    "SELECT * FROM serving_rows")
    finally:
        con.close()
    return db_path


def main() -> None:
    from pantryiq.lakehouse.catalog import get_catalog

    write_silver(build_rows(get_catalog()))
    con = duckdb.connect(str(DEFAULT_DB), read_only=True)
    try:
        total, stated = con.execute(
            "SELECT count(*), count(servings) FROM silver.recipe_servings").fetchone()
        spread = con.execute(
            "SELECT min(servings), median(servings), max(servings) "
            "FROM silver.recipe_servings WHERE servings IS NOT NULL").fetchone()
    finally:
        con.close()

    print(f"silver.recipe_servings: {total:,} recipes")
    print(f"  with a stated servings count: {stated:,} ({100 * stated / total:.1f}%)")
    print(f"  range: {spread[0]:.0f} / median {spread[1]:.0f} / {spread[2]:.0f}")
    print("  The rest stay NULL. Estimating them from total mass would put an invented")
    print("  denominator inside every per-serving number the agent quotes.")


if __name__ == "__main__":
    main()
