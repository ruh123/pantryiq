"""Normalize USDA FDC foods into Silver: the canonical entity set entity resolution targets.

`bronze.raw_usda_foods` -> `silver.usda_foods`. Three things happen here:

1. **Energy mapping.** `Energy` is reported under three different nutrient names AND under two
   units — 7,887 foods carry both `Energy [KCAL]` (208) and `Energy [kJ]` (268), so matching on
   the name alone silently picks kJ values that are 4.184x too large. We key on the nutrient
   *number*, which is unambiguous.
2. **Search text.** 98.7% of FDC descriptions are comma-inverted facet strings
   ("Egg, whole, raw"). Reversing the facets ("raw whole egg") reads much more like a recipe
   ingredient; 2.4 measures which form actually retrieves better.
3. **Deprioritization.** Babyfood/restaurant/branded entries are legitimate USDA rows but are
   almost never what a recipe line means. They are flagged, never deleted — the flag is a
   scoring feature (2.5), and the labeling guide tells the labeler to skip them.

Run directly:  uv run python -m pantryiq.silver.usda_foods
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog

DEFAULT_DB = Path("data/pantryiq.duckdb")
BRONZE_TABLE = "bronze.raw_usda_foods"

# USDA nutrient numbers. Stable identifiers — unlike the names, which vary by dataType.
KCAL_NUMBERS = ("208", "958", "957")  # plain Energy, then Atwater Specific, then General
PROTEIN_NUMBER = "203"
FAT_NUMBER = "204"
CARB_NUMBER = "205"

# Pure fats and oils (lard, beef tallow, fish oils) sit at 902 kcal/100g in FDC; nothing
# edible goes higher.
MAX_KCAL_PER_100G = 910

# First-facet categories a recipe line essentially never means.
DEPRIORITIZED_CATEGORIES = {"babyfood", "restaurant foods", "fast foods", "formula"}
# SR Legacy writes brand and restaurant names in caps ("APPLEBEE'S", "GERBER", "CREAMSICLE").
_BRAND_CAPS = re.compile(r"\b[A-Z][A-Z'&.]{3,}\b")
# FDC boilerplate on 57 staples. Its "USDA'" reads as a brand token and its words swamp the
# search text, so it is dropped — but ONLY this phrase: every other parenthetical in the
# corpus is a synonym recipes actually use ("(scallion)", "(kiwi)", "(pinto)", "(chops)").
_FDC_BOILERPLATE = re.compile(r"\s*\(includes foods for [^)]*\)", re.I)


def _facets(description: str) -> list[str]:
    description = _FDC_BOILERPLATE.sub("", description)
    return [part.strip() for part in description.split(",") if part.strip()]


def canonical_name(description: str) -> str:
    """Lowercased description with facet commas flattened to spaces."""
    return re.sub(r"\s+", " ", " ".join(_facets(description))).lower().strip()


def search_text(description: str) -> str:
    """Facets reversed so the head noun lands last, as in recipe text.

    "Egg, whole, raw, frozen" -> "frozen raw whole egg"
    """
    return re.sub(r"\s+", " ", " ".join(reversed(_facets(description)))).lower().strip()


def category(description: str) -> str:
    """Proxy category: the leading facet. The abridged FDC payload carries no foodCategory."""
    facets = _facets(description)
    return facets[0].lower() if facets else ""


def is_deprioritized(description: str) -> bool:
    if category(description) in DEPRIORITIZED_CATEGORIES:
        return True
    return bool(_BRAND_CAPS.search(_FDC_BOILERPLATE.sub("", description)))


def _nutrients(food: dict) -> dict[str, float]:
    """Nutrient number -> amount per 100g, keeping the first occurrence of each."""
    amounts: dict[str, float] = {}
    for nutrient in food.get("foodNutrients") or []:
        number, amount = nutrient.get("number"), nutrient.get("amount")
        if number is not None and amount is not None and number not in amounts:
            amounts[str(number)] = float(amount)
    return amounts


def kcal_per_100g(food: dict) -> float | None:
    """Energy in kcal, resolving the three name variants via nutrient number.

    Prefers plain `Energy` (208); falls back to Atwater Specific (958) over General (957),
    which is USDA's own accuracy ordering for the Foundation foods that carry both.
    """
    amounts = _nutrients(food)
    for number in KCAL_NUMBERS:
        if number in amounts:
            return amounts[number]
    return None


def build_usda_rows(catalog: Catalog) -> pa.Table:
    """Read Bronze USDA foods and shape them into the Silver canonical entity set."""
    bronze = catalog.load_table(BRONZE_TABLE).scan().to_arrow()
    columns: dict[str, list] = {key: [] for key in (
        "fdc_id", "data_type", "description_raw", "canonical_name", "search_text", "category",
        "kcal_per_100g", "protein_g", "fat_g", "carb_g", "is_deprioritized",
    )}
    for fdc_id, data_type, payload in zip(
        bronze.column("fdc_id").to_pylist(),
        bronze.column("data_type").to_pylist(),
        bronze.column("raw_payload").to_pylist(),
    ):
        food = json.loads(payload)
        description = food.get("description") or ""
        amounts = _nutrients(food)
        columns["fdc_id"].append(fdc_id)
        columns["data_type"].append(data_type)
        columns["description_raw"].append(description)
        columns["canonical_name"].append(canonical_name(description))
        columns["search_text"].append(search_text(description))
        columns["category"].append(category(description))
        columns["kcal_per_100g"].append(kcal_per_100g(food))
        columns["protein_g"].append(amounts.get(PROTEIN_NUMBER))
        columns["fat_g"].append(amounts.get(FAT_NUMBER))
        columns["carb_g"].append(amounts.get(CARB_NUMBER))
        columns["is_deprioritized"].append(is_deprioritized(description))
    return pa.table(columns)


def write_silver(foods: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    """Write `silver.usda_foods` to DuckDB, replacing it (idempotent re-run)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("usda_rows", foods)
        con.execute("CREATE OR REPLACE TABLE silver.usda_foods AS SELECT * FROM usda_rows")
    finally:
        con.close()
    return db_path


def main() -> None:
    from pantryiq.lakehouse.catalog import get_catalog

    foods = build_usda_rows(get_catalog())
    write_silver(foods)

    total = foods.num_rows
    kcal = [k for k in foods.column("kcal_per_100g").to_pylist() if k is not None]
    flagged = sum(foods.column("is_deprioritized").to_pylist())
    out_of_bounds = [k for k in kcal if k < 0 or k > MAX_KCAL_PER_100G]
    print(f"silver.usda_foods: {total:,} canonical entities")
    print(f"  kcal present   : {len(kcal):,} ({100 * len(kcal) / total:.1f}%)")
    print(f"  kcal range     : {min(kcal):.0f}-{max(kcal):.0f} per 100g "
          f"(out of bounds: {len(out_of_bounds)})")
    print(f"  deprioritized  : {flagged:,} ({100 * flagged / total:.1f}%)")
    print(f"  distinct categories: {len(set(foods.column('category').to_pylist())):,}")


if __name__ == "__main__":
    main()
