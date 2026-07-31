"""Convert parsed recipe quantities into grams — the step the whole Gold layer waits on.

`silver.recipe_ingredient_lines` + `silver.usda_portions` -> `silver.recipe_ingredient_grams`.
Nutrition in `silver.usda_foods` is per 100g; recipe lines say "2 cups", "1 (16 oz.) can",
"4 boned chicken breasts". Without a mass there is no per-recipe total, no `data_trust_score`,
and nothing for the agent to answer from.

Four paths, tried in order of directness. `method` records which one fired, so coverage can be
read per path rather than as one number:

1. **mass** — "1 lb.", "8 oz", "500 g". Pure arithmetic, no lookup, always right.
2. **pack size** — "2 (16 oz.) pkg. frozen corn". The parser keeps `quantity=2, unit=package`
   and drops the "16 oz", which is the only part that states a mass. Recovering it converts a
   container into an exact weight; 30.6% of package lines carry one.
3. **volumetric** — "2 cups flour" against the resolved entity's cup weight. Measures inside the
   volume family are exact ratios (4 cups to a quart), so a recipe quart still resolves against
   a USDA cup weight.
4. **count** — "1 egg", where the line names no unit at all and only a per-item portion
   ("large", "each", RACC) can price it.

**`grams` is NULL when no path applies, never 0.** A zero is a claim the ingredient weighs
nothing and would silently drag a recipe's totals down; a null is a measured gap that
`nutrition_coverage` reports honestly. Same discipline as `nutrition.error` returning None.

**Unqualified portions win.** USDA carries "cup" and "cup, sifted" separately — 125g against
100g for flour — and a recipe line saying "1 cup flour" has not told us which. Taking the plain
measure is the assumption; the qualified rows stay in Silver so the sensitivity stays testable.

Run directly:  uv run python -m pantryiq.silver.grams
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

import duckdb
import pyarrow as pa

from pantryiq.silver.ingredient_lines import _to_float

DEFAULT_DB = Path("data/pantryiq.duckdb")

# Exact definitions, not approximations: the international avoirdupois pound is 453.59237 g.
MASS_GRAMS = {"gram": 1.0, "kilogram": 1000.0, "ounce": 28.349523125, "pound": 453.59237}

# Volume measures as multiples of a US cup. These ratios are exact, so a recipe measure can be
# converted into whatever volume measure USDA happens to publish for that food — without this,
# "1 quart milk" fails against a food whose only portion is a cup.
CUP_EQUIVALENT = {
    "cup": 1.0,
    "tablespoon": 1 / 16,
    "teaspoon": 1 / 48,
    "fluid_ounce": 1 / 8,
    "pint": 2.0,
    "quart": 4.0,
    "gallon": 16.0,
    "milliliter": 1 / 236.5882365,
    "liter": 1000 / 236.5882365,
}

# Measures that price ONE of a thing, for lines that name no unit ("1 egg", "2 onions").
# RACC and NLEA servings are USDA's standard reference amounts.
COUNT_MEASURES = ("each", "piece", "large", "medium", "racc", "serving", "small")

# "1 (10 1/2 oz.) can", "2 (16 oz) pkg" — the pack size the parser discards.
_PACK_SIZE = re.compile(
    r"\(\s*([\d\s./]+?)\s*(oz|ounce|lb|pound|kg|g|gram|ml|liter|l)s?\.?\s*\)", re.I)
# "2 (16 oz.) pkg" — the parenthetical follows the leading quantity, so it sizes ONE pack.
# Anything else ("7 pt. (5 lb.) sugar") states a total or an equivalence.
_LEADING_PACK = re.compile(r"^\s*[\d\s./]+\s*\(")
_PACK_UNITS = {"oz": "ounce", "ounce": "ounce", "lb": "pound", "pound": "pound",
               "kg": "kilogram", "g": "gram", "gram": "gram"}
_PACK_VOLUME = {"ml": "milliliter", "l": "liter", "liter": "liter"}

MethodGrams = tuple[float, str] | None


def from_mass(quantity: float | None, unit: str | None) -> MethodGrams:
    """Path 1 — the unit is already a mass, so no lookup is needed."""
    if quantity is None or unit not in MASS_GRAMS:
        return None
    return (quantity * MASS_GRAMS[unit], "mass")


def from_pack_size(quantity: float | None, line_raw: str, portions: dict) -> MethodGrams:
    """Path 2 — recover the pack size the parser dropped: "2 (16 oz.) pkg" is 2 x 453.6g.

    **The parenthetical is only a PER-ITEM size when it directly follows the leading quantity.**
    Elsewhere it is an equivalence or a total — "7 pt. (5 lb.) sugar" means those 7 pints weigh
    5 lb, not 7 x 5 lb, and "10 eggs (1 lb.)" is one pound of eggs, not ten. Multiplying by
    `quantity` in those cases overstated 239 lines by a factor, 3.09x in aggregate across 218
    recipes. `_LEADING_PACK` is what distinguishes the two shapes.

    A volume pack ("1 (500 ml) carton") still needs the food's density, so it routes through the
    volumetric lookup rather than being treated as a mass.
    """
    if quantity is None:
        return None
    match = _PACK_SIZE.search(line_raw)
    if not match:
        return None
    amount = _to_float(match.group(1))
    if amount is None:
        return None
    # Anchored to the leading quantity -> per item, so multiply. Otherwise the parenthetical
    # restates the whole amount and the quantity is already accounted for inside it.
    packs = quantity if _LEADING_PACK.match(line_raw) else 1.0
    token = match.group(2).lower()
    if token in _PACK_UNITS:
        return (packs * amount * MASS_GRAMS[_PACK_UNITS[token]], "pack_size")
    volume = from_volume(packs * amount, _PACK_VOLUME.get(token), portions)
    return (volume[0], "pack_size_volume") if volume else None


def _weight_for(portions: dict, measure: str) -> float | None:
    """Grams for one `measure` of this food, preferring the unqualified variant.

    `portions` maps measure -> {qualifier or None: grams}. When only qualified rows exist there
    is no unqualified truth to prefer, so the median is taken rather than an arbitrary pick.
    """
    variants = portions.get(measure)
    if not variants:
        return None
    if None in variants:
        return variants[None]
    return statistics.median(variants.values())


def from_volume(quantity: float | None, unit: str | None, portions: dict) -> MethodGrams:
    """Path 3 — a volume measure priced against whatever volume measure USDA published.

    The food's OWN weight for this exact measure wins. Only when USDA never published it do we
    derive one from another volume measure: the ratios are exact, but the derivation is not —
    a cup of flour is 125g while a tablespoon is 7.8g, not 125/16 = 7.81g, because these are
    separately measured quantities and things pack differently at different scales.
    """
    if quantity is None or unit not in CUP_EQUIVALENT:
        return None
    exact = _weight_for(portions, unit)
    if exact is not None:
        return (quantity * exact, "volumetric")

    cups = quantity * CUP_EQUIVALENT[unit]
    for measure, per_cup in sorted(CUP_EQUIVALENT.items(), key=lambda item: -item[1]):
        weight = _weight_for(portions, measure)
        if weight is not None:
            return (cups / per_cup * weight, "volumetric_derived")
    return None


def from_count(quantity: float | None, unit: str | None, portions: dict) -> MethodGrams:
    """Path 4 — "1 egg": no unit at all, so only a per-item portion can price it."""
    if quantity is None or unit is not None:
        return None
    for measure in COUNT_MEASURES:
        weight = _weight_for(portions, measure)
        if weight is not None:
            return (quantity * weight, "count")
    return None


def convert(quantity: float | None, unit: str | None, line_raw: str,
            portions: dict) -> MethodGrams:
    """The first path that applies, or None. `portions` is this line's resolved entity's."""
    if quantity is None:
        return None
    # Where the pack size sits in the order depends on what the line's own unit is:
    #
    # - a CONTAINER ("2 (16 oz.) pkg") — the parenthetical is the only statement of mass, so it
    #   comes first;
    # - a VOLUME ("7 pt. (5 lb.) sugar") — the food's own density is the direct reading, but a
    #   stated mass beats no answer at all, so the parenthetical is a fallback behind it.
    #
    # It never precedes a volume unit, because taking "2 c. (8 oz.)" as 2 x 8 oz double-counted.
    pack = from_pack_size(quantity, line_raw, portions)
    container = unit not in MASS_GRAMS and unit not in CUP_EQUIVALENT
    for candidate in (from_mass(quantity, unit),
                      pack if container else None,
                      from_volume(quantity, unit, portions),
                      None if container else pack,
                      from_count(quantity, unit, portions)):
        if candidate is not None and candidate[0] > 0:
            return candidate
    return None


def portion_index(db_path: Path | str = DEFAULT_DB) -> dict[str, dict]:
    """{fdc_id: {measure: {qualifier: grams}}} for every food with a usable portion."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT fdc_id, measure, qualifier, gram_weight FROM silver.usda_portions").fetchall()
    finally:
        con.close()
    index: dict[str, dict] = {}
    for fdc_id, measure, qualifier, grams in rows:
        index.setdefault(fdc_id, {}).setdefault(measure, {})[qualifier] = float(grams)
    return index


def build_rows(db_path: Path | str = DEFAULT_DB) -> pa.Table:
    """One row per ingredient line, with grams where a path applied.

    Lines whose string the resolver DECLINED (§12) get no entity and therefore no portions, so
    they convert only by mass or pack size — which is correct: we do not know what the food is,
    so we must not price it by someone else's density.
    """
    index = portion_index(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        lines = con.execute(
            """
            SELECT l.recipe_id, l.line_index, l.line_raw, l.quantity, l.unit,
                   CASE WHEN m.abstained THEN NULL ELSE m.fdc_id END AS fdc_id
            FROM silver.recipe_ingredient_lines l
            LEFT JOIN silver.ingredient_entity_map m ON m.normalized_text = l.normalized_text
            ORDER BY l.recipe_id, l.line_index
            """
        ).fetchall()
    finally:
        con.close()

    recipe_ids, line_indexes, grams, methods = [], [], [], []
    for recipe_id, line_index, line_raw, quantity, unit, fdc_id in lines:
        result = convert(quantity, unit, line_raw or "", index.get(fdc_id or "", {}))
        recipe_ids.append(recipe_id)
        line_indexes.append(line_index)
        grams.append(result[0] if result else None)
        methods.append(result[1] if result else None)

    return pa.table({
        "recipe_id": recipe_ids,
        "line_index": line_indexes,
        "grams": grams,
        "method": methods,
        "converted": [value is not None for value in grams],
    })


def write_silver(rows: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    """Write `silver.recipe_ingredient_grams`, replacing it (idempotent re-run)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("gram_rows", rows)
        con.execute("CREATE OR REPLACE TABLE silver.recipe_ingredient_grams AS "
                    "SELECT * FROM gram_rows")
    finally:
        con.close()
    return db_path


def main() -> None:
    write_silver(build_rows())
    con = duckdb.connect(str(DEFAULT_DB), read_only=True)
    try:
        total, converted = con.execute(
            "SELECT count(*), sum(converted::int) FROM silver.recipe_ingredient_grams").fetchone()
        by_method = con.execute(
            "SELECT coalesce(method, '(none)'), count(*) FROM silver.recipe_ingredient_grams "
            "GROUP BY 1 ORDER BY 2 DESC").fetchall()
        by_class = con.execute(
            """
            SELECT CASE
                     WHEN l.quantity IS NULL THEN 'no quantity'
                     WHEN l.unit IS NULL THEN 'count (no unit)'
                     WHEN l.unit IN ('gram','kilogram','ounce','pound') THEN 'mass'
                     WHEN l.unit IN ('cup','teaspoon','tablespoon','quart','pint','gallon')
                       THEN 'volumetric'
                     ELSE 'package/other' END AS unit_class,
                   count(*), sum(g.converted::int)
            FROM silver.recipe_ingredient_lines l
            JOIN silver.recipe_ingredient_grams g
              ON g.recipe_id = l.recipe_id AND g.line_index = l.line_index
            GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
        recipes = con.execute(
            """
            SELECT median(share) FROM (
              SELECT sum(converted::int)::double / count(*) AS share
              FROM silver.recipe_ingredient_grams GROUP BY recipe_id)
            """
        ).fetchone()[0]
    finally:
        con.close()

    print("=" * 78)
    print("3.3 — silver.recipe_ingredient_grams")
    print("=" * 78)
    print(f"  {converted:,} of {total:,} ingredient lines converted to grams "
          f"({100 * converted / total:.1f}%)")
    print(f"  median share of a recipe's lines converted: {recipes:.1%}")
    print("\n  by path")
    for method, count in by_method:
        print(f"    {method:20} {count:7,}  ({100 * count / total:5.1f}%)")
    print("\n  by unit class")
    for unit_class, count, hit in by_class:
        print(f"    {unit_class:20} {hit:7,} / {count:7,}  ({100 * hit / count:5.1f}%)")
    print("\n  Uncovered lines are NULL, not 0 — a null is a measured gap; a zero would claim")
    print("  the ingredient weighs nothing and silently drag the recipe's totals down.")


if __name__ == "__main__":
    main()
