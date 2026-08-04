"""A miniature warehouse, so the rules that matter run without the 15,000-recipe corpus.

22 of the suite's tests needed `data/pantryiq_gold.duckdb` or `data/pantryiq.duckdb` and were
skipped everywhere else — **including all nine dietary-tag tests**, the file this project calls
"the one thing that could hurt someone if it is wrong". A safety check that only runs on the
author's laptop is close to no check at all.

This builds Silver from scratch and lets **dbt build the real Gold models on top of it**. That
distinction is the point: asserting against tags this module wrote would be a tautology, so the
tags are produced by `models/gold/recipe_tags.sql` exactly as in production, from ingredients
whose right answer is known by construction.

Ten recipes, each existing to exercise one branch of the rule:

    r_vegan       onion + salt                     -> vegetarian, vegan, gluten-free
    r_dairy       onion + milk                     -> vegetarian, gluten-free (milk is not vegan)
    r_wheat       onion + wheat flour              -> vegetarian, vegan (flour is not GF)
    r_meat        onion + ground beef              -> gluten-free only
    r_worcester   onion + worcestershire           -> nothing: anchovies, the case that broke
                                                     both previous rules
    r_unknown     onion + "Sauce, unspecified"     -> nothing: `unknown` disqualifies
    r_truncated   onion + salt, method names       -> gluten-free only: the directions veto,
                  sausage                             for a corpus that omits an ingredient
    r_millet      a line reading "flour" that      -> vegetarian, vegan, NOT gluten-free:
                  resolved to *Millet flour*          the entity is GF and the recipe is not
    r_partial     onion + an unresolved line       -> nothing: coverage below 1.0
    r_nameflag    onion + "chicken broth" resolved -> gluten-free only: the name denylist
                  to a harmless entity                catching entity resolution being wrong
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import duckdb

# Real USDA fdc_ids, chosen because `seeds/cost_reference.csv` prices them — that makes every
# fixture recipe fully costed, so `cost_coverage` reaches 1.0 and the planner's candidate filter
# has something to select. Synthetic ids would join to no price and the planner would see an
# empty pool.
PRICED = ['746784', '169655', '168833', '169656', '169761', '168895', '172023', '171287', '170000']

# (fdc_id, description, category, kcal, protein, fat, carb, vegetarian, vegan, gluten_free)
ENTITIES = [
    (PRICED[0], "Onions, raw", "Vegetables and Vegetable Products", 40.0, 1.1, 0.1, 9.3,
     "yes", "yes", "yes"),
    (PRICED[1], "Wheat flour, white, all-purpose", "Cereal Grains and Pasta", 364.0, 10.3, 1.0, 76.3,
     "yes", "yes", "no"),
    (PRICED[2], "Sauce, worcestershire", "Soups, Sauces, and Gravies", 78.0, 0.0, 0.0, 19.0,
     "no", "no", "unknown"),
    (PRICED[3], "Beef, ground, raw", "Beef Products", 254.0, 17.2, 20.0, 0.0,
     "no", "no", "yes"),
    (PRICED[4], "Milk, whole, 3.25% milkfat", "Dairy and Egg Products", 61.0, 3.2, 3.3, 4.8,
     "yes", "no", "yes"),
    (PRICED[5], "Sauce, unspecified", None, 100.0, 1.0, 5.0, 10.0,
     "unknown", "unknown", "unknown"),
    (PRICED[6], "Millet flour", "Cereal Grains and Pasta", 382.0, 10.8, 4.2, 75.1,
     "yes", "yes", "yes"),
    (PRICED[7], "Salt, table", "Spices and Herbs", 0.0, 0.0, 0.0, 0.0,
     "yes", "yes", "yes"),
    (PRICED[8], "Vinegar, cider", "Spices and Herbs", 21.0, 0.0, 0.0, 0.9,
     "yes", "yes", "yes"),
]

# (recipe_id, title, directions, [(line_text, fdc_id_or_None), ...])
RECIPES = [
    ("r_vegan", "Onion Relish", "combine and chill.",
     [("onion", PRICED[0]), ("salt", PRICED[7])]),
    ("r_dairy", "Creamed Onions", "warm gently.",
     [("onion", PRICED[0]), ("milk", PRICED[4])]),
    ("r_wheat", "Onion Fritters", "fry until golden.",
     [("onion", PRICED[0]), ("all-purpose flour", PRICED[1])]),
    ("r_meat", "Beef And Onion", "brown well.",
     [("onion", PRICED[0]), ("ground beef", PRICED[3])]),
    ("r_worcester", "Onion Dip", "stir together.",
     [("onion", PRICED[0]), ("worcestershire sauce", PRICED[2])]),
    ("r_unknown", "Mystery Onions", "combine.",
     [("onion", PRICED[0]), ("sauce", PRICED[5])]),
    ("r_truncated", "Pickled Sausage", "remove skin from 2 rings sausage and slice.",
     [("onion", PRICED[0]), ("vinegar", PRICED[8])]),
    ("r_millet", "Icebox Cookies", "chill and slice.",
     [("flour", PRICED[6]), ("salt", PRICED[7])]),
    ("r_partial", "Half Known", "mix.",
     [("onion", PRICED[0]), ("something nobody resolved", None)]),
    ("r_nameflag", "Broth Onions", "simmer.",
     [("onion", PRICED[0]), ("chicken broth", PRICED[8])]),
]

# Distinct counts so a ranking is observable. "onion" is deliberately the most frequent and
# "sauce" the least, so the top of a DESC ranking is unambiguous.
OCCURRENCES = {
    "onion": 91, "all-purpose flour": 74, "milk": 63, "ground beef": 52,
    "worcestershire sauce": 41, "flour": 33, "salt": 22, "vinegar": 14,
    "chicken broth": 8, "something nobody resolved": 3, "sauce": 1,
}

# Resolved, but marked for review — the flagged-but-not-abstained case, which is 157 of the
# 2,884 undecided strings in the real warehouse and was represented by nothing here.
FLAGGED_ONLY = {"worcestershire sauce"}

SILVER_DDL = """
CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE TABLE silver.usda_foods (
    fdc_id VARCHAR, data_type VARCHAR, description_raw VARCHAR, canonical_name VARCHAR,
    search_text VARCHAR, category VARCHAR, kcal_per_100g DOUBLE, protein_g DOUBLE,
    fat_g DOUBLE, carb_g DOUBLE, is_deprioritized BOOLEAN);

CREATE OR REPLACE TABLE silver.usda_food_categories (fdc_id VARCHAR, food_category VARCHAR);

CREATE OR REPLACE TABLE silver.entity_dietary_flags (
    fdc_id VARCHAR, is_vegetarian VARCHAR, is_vegan VARCHAR, is_gluten_free VARCHAR,
    reason VARCHAR);

CREATE OR REPLACE TABLE silver.recipe_ingredient_lines (
    recipe_id VARCHAR, line_index BIGINT, line_raw VARCHAR, quantity DOUBLE, unit VARCHAR,
    ingredient_text VARCHAR, normalized_text VARCHAR);

CREATE OR REPLACE TABLE silver.recipe_ingredient_grams (
    recipe_id VARCHAR, line_index BIGINT, grams DOUBLE, method VARCHAR, converted BOOLEAN);

CREATE OR REPLACE TABLE silver.ingredient_entity_map (
    normalized_text VARCHAR, occurrence_count BIGINT, frequency_stratum VARCHAR,
    fdc_id VARCHAR, description_raw VARCHAR, kcal_per_100g DOUBLE, protein_g DOUBLE,
    fat_g DOUBLE, carb_g DOUBLE, is_deprioritized BOOLEAN, cosine DOUBLE, confidence DOUBLE,
    flagged BOOLEAN, abstained BOOLEAN);

CREATE OR REPLACE TABLE silver.distinct_ingredient_strings (
    normalized_text VARCHAR, occurrence_count BIGINT, frequency_stratum VARCHAR);

CREATE OR REPLACE TABLE silver.recipe_servings (recipe_id VARCHAR, servings BIGINT);

CREATE OR REPLACE TABLE silver.recipe_meta (
    recipe_id VARCHAR, title VARCHAR, source_url VARCHAR, directions VARCHAR);
"""


def build_silver(db_path: Path | str) -> Path:
    """Create a complete synthetic Silver at `db_path`."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(SILVER_DDL)

        for fdc_id, description, group, kcal, protein, fat, carb, veg, vegan, gf in ENTITIES:
            con.execute("INSERT INTO silver.usda_foods VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        [fdc_id, "sr_legacy", description, description.lower(),
                         description.lower(), description.split(",")[0].lower(),
                         kcal, protein, fat, carb, False])
            con.execute("INSERT INTO silver.usda_food_categories VALUES (?,?)", [fdc_id, group])
            con.execute("INSERT INTO silver.entity_dietary_flags VALUES (?,?,?,?,?)",
                        [fdc_id, veg, vegan, gf, None])

        seen: set[str] = set()
        for recipe_id, title, directions, lines in RECIPES:
            con.execute("INSERT INTO silver.recipe_meta VALUES (?,?,?,?)",
                        [recipe_id, title, "example.com/" + recipe_id, directions])
            con.execute("INSERT INTO silver.recipe_servings VALUES (?,?)", [recipe_id, 4])
            for index, (text, fdc_id) in enumerate(lines):
                con.execute("INSERT INTO silver.recipe_ingredient_lines VALUES (?,?,?,?,?,?,?)",
                            [recipe_id, index, f"1 c. {text}", 1.0, "cup", text, text])
                # Every line is weighed; whether it CONTRIBUTES depends on having an entity, so
                # `r_partial` lands below full coverage exactly as an unresolved line does live.
                con.execute("INSERT INTO silver.recipe_ingredient_grams VALUES (?,?,?,?,?)",
                            [recipe_id, index, 100.0, "volumetric", True])
                if text not in seen:
                    seen.add(text)
                    # Occurrence counts must VARY, or a ranking test cannot discriminate: with
                    # every string at 10, reversing `ORDER BY occurrence_count DESC` produced an
                    # identical list and the mutation survived.
                    occurrences = OCCURRENCES.get(text, 5)
                    con.execute("INSERT INTO silver.distinct_ingredient_strings VALUES (?,?,?)",
                                [text, occurrences, "head"])
                    entity = next((e for e in ENTITIES if e[0] == fdc_id), None)
                    con.execute(
                        "INSERT INTO silver.ingredient_entity_map VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [text, occurrences, "head", fdc_id,
                         entity[1] if entity else None,
                         entity[3] if entity else None,
                         entity[4] if entity else None,
                         entity[5] if entity else None,
                         entity[6] if entity else None,
                         False, 0.9 if fdc_id else 0.2, 0.9 if fdc_id else 0.1,
                         # flagged and abstained must DIFFER on at least one row, or
                         # `WHERE abstained OR flagged` and `... AND ...` return the same set and
                         # the difference between them is untestable.
                         text in FLAGGED_ONLY or fdc_id is None, fdc_id is None])
    finally:
        con.close()
    return db_path


def build_gold(db_path: Path | str, project_root: Path | str = ".") -> Path:
    """Run the REAL dbt models against the synthetic Silver.

    Builds into `gold_staging` via the existing `staging` target — the same path the production
    gate uses — so the tags under test come from `models/gold/recipe_tags.sql` and not from this
    file. `PANTRYIQ_DB` already parameterises the warehouse location, so no profile change is
    needed.
    """
    environment = {**os.environ, "PANTRYIQ_DB": str(Path(db_path).resolve())}
    result = subprocess.run(
        ["uv", "run", "dbt", "build", "--profiles-dir", ".", "--target", "staging"],
        cwd=str(project_root), capture_output=True, text=True, env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dbt build against the fixture warehouse failed:\n{result.stdout[-3000:]}")
    return Path(db_path)
