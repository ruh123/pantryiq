"""Normalize USDA `foodPortions` into Silver: grams per (food, measure).

`bronze.raw_usda_portions` -> `silver.usda_portions`. This is the lookup that turns "2 cups
flour" into a mass, and three things about the source shape drive the parsing:

1. **The measure is in `modifier`, not `measureUnit`.** On SR Legacy — 95% of the canonical set —
   `measureUnit.name` is the literal string `"undetermined"` and the real measure sits in
   `modifier` ("cup", "tsp", "cup, chopped"). Foundation is the other way round: `measureUnit`
   carries `RACC`/`milliliter` and `modifier` is null. Reading `measureUnit` first would discard
   almost every usable row, so `modifier` wins and `measureUnit` is the fallback.

2. **`gramWeight` is for `amount` units, not for one.** A row reading amount=2, modifier="tbsp",
   gramWeight=28.4 means 2 tbsp weigh 28.4g. Dividing by `amount` is what makes the column mean
   "grams per one measure"; skipping it silently doubles or halves those foods.

3. **`fl oz` is not `oz`.** One is volume, the other mass, and the recipe-side `UNITS` map sends
   the bare token "oz" to `ounce`. They are kept distinct here, so a fluid-ounce portion simply
   fails to join rather than joining wrongly — an honest gap beats a confident 28.35g.

Qualified and unqualified variants are BOTH kept ("cup" and "cup, sifted" are separate rows,
100g vs 125g for flour). Silver preserves them; picking one is the consumer's job, so the
sensitivity is measurable rather than baked in here.

Run directly:  uv run python -m pantryiq.silver.usda_portions
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog

from pantryiq.silver.ingredient_lines import UNITS

DEFAULT_DB = Path("data/pantryiq.duckdb")
BRONZE_TABLE = "bronze.raw_usda_portions"

# Measures whose first token is misleading on its own. Checked before the single-token lookup:
# "fl oz" would otherwise tokenize to "fl" (no match) or be read as "oz" -> ounce, conflating a
# volume with a mass.
MULTIWORD_MEASURES = {
    "fl oz": "fluid_ounce",
    "fluid ounce": "fluid_ounce",
    "fl. oz": "fluid_ounce",
    "cubic inch": "cubic_inch",
    "nlea serving": "serving",
}
# USDA-specific measures with no recipe-side equivalent. Kept verbatim so they are available for
# the count/serving fallback in 3.3 rather than silently dropped.
PASSTHROUGH_MEASURES = {"racc", "serving", "guideline amount", "large", "medium", "small"}

_PAREN = re.compile(r"\([^)]*\)")
_UNDETERMINED = {"", "undetermined", "none", "null"}


def parse_measure(modifier: str | None, measure_unit: str | None) -> tuple[str | None, str | None]:
    """(canonical measure, qualifier) from a portion's modifier / measureUnit.

    `modifier` is preferred — see the module docstring. The qualifier is whatever narrows the
    measure ("cup, sifted" -> sifted), kept because it is worth 25% on flour.
    """
    source = (modifier or "").strip().lower()
    if source in _UNDETERMINED:
        source = (measure_unit or "").strip().lower()
    if source in _UNDETERMINED:
        return (None, None)

    # A trailing parenthetical describes the measure, it is not part of it:
    # "oz (23 whole kernels)", "can (10.75 oz)", 'pat (1" sq, 1/3" high)'.
    parenthetical = " ".join(match.strip("()") for match in _PAREN.findall(source)).strip()
    source = _PAREN.sub(" ", source).strip()

    head, _, tail = source.partition(",")
    qualifier_parts = [part.strip() for part in (tail, parenthetical) if part.strip()]
    head = head.strip()

    for phrase, canonical in MULTIWORD_MEASURES.items():
        if head == phrase or head.startswith(phrase + " "):
            trailing = head[len(phrase):].strip()
            if trailing:
                qualifier_parts.insert(0, trailing)
            return (canonical, ", ".join(qualifier_parts) or None)

    tokens = head.split()
    if not tokens:
        return (None, ", ".join(qualifier_parts) or None)

    # "cup packed" / "cup sifted" — a space-separated qualifier rather than a comma'd one.
    first, rest = tokens[0], " ".join(tokens[1:])
    if first in UNITS and (not rest or rest not in UNITS):
        if rest:
            qualifier_parts.insert(0, rest)
        return (UNITS[first], ", ".join(qualifier_parts) or None)

    if head in PASSTHROUGH_MEASURES:
        return (head, ", ".join(qualifier_parts) or None)
    return (head, ", ".join(qualifier_parts) or None)


def grams_per_unit(portion: dict) -> float | None:
    """`gramWeight` normalized to ONE measure. None when the row cannot be trusted.

    `amount` defaults to 1 when absent, but a present-and-zero amount is unusable rather than
    a division by zero, and a non-positive gram weight is not a weight.
    """
    grams = portion.get("gramWeight")
    if grams is None:
        return None
    amount = portion.get("amount")
    amount = 1.0 if amount in (None, "") else float(amount)
    if amount <= 0 or float(grams) <= 0:
        return None
    return float(grams) / amount


def build_rows(catalog: Catalog) -> pa.Table:
    """Read Bronze portions and produce one row per (fdc_id, measure, qualifier)."""
    bronze = catalog.load_table(BRONZE_TABLE).scan().to_arrow()
    fdc_ids: list[str] = []
    measures: list[str | None] = []
    qualifiers: list[str | None] = []
    weights: list[float | None] = []

    for fdc_id, payload in zip(bronze.column("fdc_id").to_pylist(),
                               bronze.column("raw_payload").to_pylist()):
        for portion in json.loads(payload):
            measure, qualifier = parse_measure(
                portion.get("modifier"), (portion.get("measureUnit") or {}).get("name"))
            grams = grams_per_unit(portion)
            if measure is None or grams is None:
                continue
            fdc_ids.append(str(fdc_id))
            measures.append(measure)
            qualifiers.append(qualifier)
            weights.append(grams)

    return pa.table({
        "fdc_id": fdc_ids,
        "measure": measures,
        "qualifier": qualifiers,
        "gram_weight": weights,
    })


def write_silver(portions: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    """Write `silver.usda_portions` to DuckDB, replacing it (idempotent re-run).

    Duplicates are collapsed to the MEDIAN gram weight per (fdc_id, measure, qualifier): USDA
    sometimes lists the same measure twice with different weights, and a median is not dragged
    by one outlier the way a mean is.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("portion_rows", portions)
        con.execute(
            """
            CREATE OR REPLACE TABLE silver.usda_portions AS
            SELECT fdc_id, measure, qualifier, median(gram_weight) AS gram_weight
            FROM portion_rows
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
            """
        )
    finally:
        con.close()
    return db_path


def main() -> None:
    from pantryiq.lakehouse.writelock import warehouse_lock

    with warehouse_lock():
        from pantryiq.lakehouse.catalog import get_catalog

        write_silver(build_rows(get_catalog()))
        con = duckdb.connect(str(DEFAULT_DB), read_only=True)
        try:
            total, foods = con.execute(
                "SELECT count(*), count(DISTINCT fdc_id) FROM silver.usda_portions").fetchone()
            volumetric = con.execute(
                "SELECT count(DISTINCT fdc_id) FROM silver.usda_portions "
                "WHERE measure IN ('cup', 'tablespoon', 'teaspoon')").fetchone()[0]
            picked = con.execute(
                """
                SELECT count(DISTINCT m.fdc_id) FROM silver.ingredient_entity_map m
                JOIN silver.usda_portions p USING (fdc_id)
                WHERE NOT m.abstained AND p.measure IN ('cup', 'tablespoon', 'teaspoon')
                """).fetchone()[0]
            resolver_entities = con.execute(
                "SELECT count(DISTINCT fdc_id) FROM silver.ingredient_entity_map "
                "WHERE NOT abstained").fetchone()[0]
            by_measure = con.execute(
                "SELECT measure, count(*) FROM silver.usda_portions "
                "GROUP BY 1 ORDER BY 2 DESC LIMIT 12").fetchall()
        finally:
            con.close()

        print(f"silver.usda_portions: {total:,} rows over {foods:,} foods")
        print(f"  with a cup/tbsp/tsp weight : {volumetric:,} foods "
              f"({100 * volumetric / foods:.1f}% of foods with any portion)")
        print(f"  of the entities the resolver actually picks: {picked:,} / {resolver_entities:,} "
              f"({100 * picked / resolver_entities:.1f}%)")
        print("  most common measures:")
        for measure, count in by_measure:
            print(f"    {measure:16} {count:6,}")


if __name__ == "__main__":
    main()
