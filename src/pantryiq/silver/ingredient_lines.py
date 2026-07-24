"""Parse RecipeNLG ingredient lines into Silver: quantity, unit, ingredient text.

`bronze.raw_recipes` -> `silver.recipe_ingredient_lines` (one row per line) and
`silver.distinct_ingredient_strings` (the deduped ER working set, with occurrence counts
and a frequency stratum). Entity resolution runs on the DISTINCT strings — 112K lines
collapse to ~13K distinct, so every downstream step (embeddings, scoring, LLM) is ~8x cheaper.

Normalization is deliberately conservative: it strips preparation words ("chopped", "finely")
but KEEPS words that change the food nutritionally ("ground", "whole", "dried", "unsalted").
The filler list's effect on retrieval is measured in 2.4 rather than tuned by guesswork.

Run directly:  uv run python -m pantryiq.silver.ingredient_lines
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import NamedTuple

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog

DEFAULT_DB = Path("data/pantryiq.duckdb")
BRONZE_TABLE = "bronze.raw_recipes"

# Occurrence-count thresholds for the labeling strata (2.3a). Chosen so each stratum holds
# enough distinct strings to sample ~167 labels from; verified against the real distribution.
HEAD_MIN = 20
MID_MIN = 3

_FRACTIONS = {
    "¼": 0.25, "½": 0.5, "¾": 0.75, "⅐": 1 / 7, "⅑": 1 / 9, "⅒": 0.1,
    "⅓": 1 / 3, "⅔": 2 / 3, "⅕": 0.2, "⅖": 0.4, "⅗": 0.6, "⅘": 0.8,
    "⅙": 1 / 6, "⅚": 5 / 6, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}
_NUMBER_WORDS = {
    "a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0, "eleven": 11.0,
    "twelve": 12.0, "half": 0.5, "dozen": 12.0,
}

# alias -> canonical unit. Abbreviations dominate RecipeNLG ("c." 36.9K, "tsp." 17.3K).
UNITS = {
    "c": "cup", "cup": "cup", "cups": "cup",
    "tsp": "teaspoon", "t": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
    "tbsp": "tablespoon", "tbs": "tablespoon", "tb": "tablespoon", "tablespoon": "tablespoon",
    "tablespoons": "tablespoon",
    "oz": "ounce", "ounce": "ounce", "ounces": "ounce",
    "lb": "pound", "lbs": "pound", "pound": "pound", "pounds": "pound",
    "qt": "quart", "quart": "quart", "quarts": "quart",
    "pt": "pint", "pint": "pint", "pints": "pint",
    "gal": "gallon", "gallon": "gallon", "gallons": "gallon",
    "g": "gram", "gram": "gram", "grams": "gram", "kg": "kilogram",
    "ml": "milliliter", "l": "liter", "liter": "liter", "liters": "liter",
    "pkg": "package", "package": "package", "packages": "package", "pkgs": "package",
    "can": "can", "cans": "can", "jar": "jar", "jars": "jar",
    "bottle": "bottle", "bottles": "bottle", "box": "box", "boxes": "box",
    "bag": "bag", "bags": "bag", "envelope": "envelope", "envelopes": "envelope",
    "carton": "carton", "cartons": "carton", "container": "container", "containers": "container",
    "stick": "stick", "sticks": "stick", "clove": "clove", "cloves": "clove",
    "slice": "slice", "slices": "slice", "head": "head", "heads": "head",
    "bunch": "bunch", "bunches": "bunch", "stalk": "stalk", "stalks": "stalk",
    "sprig": "sprig", "sprigs": "sprig", "piece": "piece", "pieces": "piece",
    "dash": "dash", "dashes": "dash", "pinch": "pinch", "pinches": "pinch",
    "drop": "drop", "drops": "drop", "doz": "dozen", "dozen": "dozen",
}

# Size qualifiers sit between the quantity and the unit ("1 large jar Cheez Whiz") or
# directly before the food ("1 large onion"). Skipped when looking for a unit; stripped
# during normalization.
SIZE_WORDS = {"large", "small", "medium", "med", "jumbo", "extra", "lg", "sm"}

# Preparation/noise words only. Anything that changes the food itself — ground, whole, dried,
# dry, frozen, canned, cooked, raw, sweetened, unsalted, skim, condensed, evaporated — is
# deliberately absent, because it changes which USDA entity is correct.
FILLER = SIZE_WORDS | {
    "chopped", "finely", "coarsely", "thinly", "freshly", "fresh", "minced", "diced",
    "sliced", "shredded", "grated", "beaten", "melted", "softened", "divided", "packed",
    "firmly", "lightly", "peeled", "seeded", "stemmed", "drained", "rinsed", "washed",
    "cleaned", "crushed", "cubed", "quartered", "halved", "trimmed", "cut", "into",
    "optional", "well", "very", "about", "approximately", "plus", "more", "needed",
    "taste", "room", "temperature", "such", "as", "each", "any", "your", "favorite",
    "or",  # only survives the " or " cut on the fallback paths ("crushed or chunk pineapple")
}

# "pound cake" is a food, not a measure; every other unit word leading a normalized string
# was a leaked container unit when checked against the real corpus.
_STRIPPABLE_UNITS = set(UNITS) - {"pound"}

_SINGULAR_EXCEPTIONS = {
    "molasses", "asparagus", "hummus", "couscous", "swiss", "watercress", "cress",
    "bass", "grits",
}
# -ves plurals need the f back ("celery leaves" -> "celery leaf", not "leave"), but the
# generic rule would break "olives"/"chives", so the irregulars are listed explicitly.
_IRREGULAR_PLURALS = {"leaves": "leaf", "halves": "half", "loaves": "loaf", "knives": "knife"}

_FRAC_CHARS = "".join(_FRACTIONS)
_NUM = rf"\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\.\d+|\d+\s*[{_FRAC_CHARS}]|[{_FRAC_CHARS}]|\d+"
# A leading quantity, optionally a range ("2 or 3", "1 to 2") of which we keep the first.
_LEAD_QTY = re.compile(rf"^\s*({_NUM})(?:\s*(?:-|–|to|or)\s*(?:{_NUM}))?\s*", re.I)
# "2% milk" / "3.5% milk" state a fat content, not an amount. This has to be a separate
# pre-check: an inline (?!\s*%) lookahead is defeated by backtracking, which quietly reads
# "3.5% milk" as quantity 3 with ".5% milk" left over.
_LEAD_PERCENT = re.compile(r"^\s*\d+(?:\.\d+)?\s*%")
_LEAD_WORD_QTY = re.compile(rf"^\s*({'|'.join(_NUMBER_WORDS)})\b\s*", re.I)
_LEAD_PAREN = re.compile(r"^\s*\([^)]*\)\s*")
_PAREN = re.compile(r"\([^)]*\)")
_UNIT_TOKEN = re.compile(r"^\s*([A-Za-z]+)\.?\s*")

class ParsedLine(NamedTuple):
    quantity: float | None
    unit: str | None
    ingredient_text: str


def parse_list(cell: str) -> list | None:
    """Parse a stringified list cell (JSON first, then Python literal).

    Mirrors `scripts/inspect_recipenlg.py`: RecipeNLG cells are JSON, but the literal_eval
    fallback covers the single-quoted rows that appear in the export.
    """
    for fn in (json.loads, ast.literal_eval):
        try:
            value = fn(cell)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list):
            return value
    return None


def _to_float(token: str) -> float | None:
    """Convert one quantity token ('2 1/2', '1/2', '1.5', '½', '1½') to a float."""
    token = token.strip()
    total = 0.0
    for char in token:
        if char in _FRACTIONS:
            total += _FRACTIONS[char]
    token = "".join(c for c in token if c not in _FRACTIONS).strip()
    for part in token.split():
        try:
            total += float(part) if "/" not in part else _fraction(part)
        except (ValueError, ZeroDivisionError):
            return None
    return total or None


def _fraction(part: str) -> float:
    numerator, denominator = part.split("/", 1)
    return float(numerator) / float(denominator)


def parse_line(line: str) -> ParsedLine:
    """Split one raw ingredient line into (quantity, unit, ingredient_text)."""
    rest = line.strip()
    quantity = None

    if not _LEAD_PERCENT.match(rest):
        match = _LEAD_QTY.match(rest)
        if match:
            quantity = _to_float(match.group(1))
            rest = rest[match.end():]
        else:
            match = _LEAD_WORD_QTY.match(rest)
            if match:
                quantity = _NUMBER_WORDS[match.group(1).lower()]
                rest = rest[match.end():]

    # "1 (8 oz.) can tomato sauce" — the pack size is metadata, not the ingredient.
    rest = _LEAD_PAREN.sub("", rest)

    unit = None
    candidate = rest
    # Skip size qualifiers so "1 large jar Cheez Whiz" still finds "jar".
    while (token := _UNIT_TOKEN.match(candidate)) and token.group(1).lower() in SIZE_WORDS:
        candidate = candidate[token.end():]
    token = _UNIT_TOKEN.match(candidate)
    # Only consume the token as a unit if something is left to be the ingredient — a line
    # reading just "cloves" names the food, not the measure.
    if token and token.group(1).lower() in UNITS and candidate[token.end():].strip():
        unit = UNITS[token.group(1).lower()]
        rest = candidate[token.end():]

    return ParsedLine(quantity, unit, rest.strip())


def singularize(word: str) -> str:
    """Crude, dependency-free singularization of the final noun."""
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if word in _SINGULAR_EXCEPTIONS or len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("oes", "shes", "ches", "xes", "sses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _reduce(text: str) -> str:
    text = re.sub(r"[^a-z0-9%\- ]", " ", text)
    # Drop stray counts ("juice of 1 lemon" -> "juice of lemon") but keep fat content ("2% milk").
    text = re.sub(r"\b\d+(\.\d+)?\b(?!\s*%)", " ", text)

    words = [w for w in text.split() if w and w not in FILLER]
    while words and words[0] == "of":
        words.pop(0)  # "dash of pepper" -> "pepper"
    # A chained second unit survives parse_line ("6 oz. can lemon juice" -> "can lemon juice").
    # Left in, it splits the ER working set: "tsp salt" and "salt" become separate entities.
    while len(words) > 1 and words[0] in _STRIPPABLE_UNITS:
        words.pop(0)
    if words:
        words[-1] = singularize(words[-1])
    return " ".join(words).strip()


def normalize(text: str) -> str:
    """Reduce parsed ingredient text to the canonical string entity resolution matches on.

    Cutting at the first comma ("morsels, divided") and at " or " ("butter or margarine")
    usually isolates the ingredient, but both cuts can swallow it whole — "drained, crushed
    pineapple", "crushed or chunk pineapple". So try the cuts hardest-first and keep the first
    one that leaves something behind.
    """
    text = _PAREN.sub(" ", text.lower())
    before_comma = text.split(",")[0]
    for candidate in (re.split(r"\bor\b", before_comma)[0], before_comma, text.replace(",", " ")):
        if reduced := _reduce(candidate):
            return reduced
    return ""


def stratum(occurrence_count: int) -> str:
    """Frequency stratum used to sample gold labels (2.3a) and report per-stratum metrics."""
    if occurrence_count >= HEAD_MIN:
        return "head"
    if occurrence_count >= MID_MIN:
        return "mid"
    return "tail"


def build_line_rows(catalog: Catalog) -> pa.Table:
    """Read Bronze recipes and produce one parsed row per ingredient line."""
    bronze = catalog.load_table(BRONZE_TABLE).scan().to_arrow()
    recipe_ids, indexes, raws, quantities, units, texts, normals = [], [], [], [], [], [], []
    for recipe_id, payload in zip(
        bronze.column("recipe_id").to_pylist(), bronze.column("raw_payload").to_pylist()
    ):
        lines = parse_list(json.loads(payload).get("ingredients", "")) or []
        for index, line in enumerate(lines):
            if not isinstance(line, str) or not line.strip():
                continue
            parsed = parse_line(line)
            recipe_ids.append(recipe_id)
            indexes.append(index)
            raws.append(line.strip())
            quantities.append(parsed.quantity)
            units.append(parsed.unit)
            texts.append(parsed.ingredient_text)
            normals.append(normalize(parsed.ingredient_text))
    return pa.table(
        {
            "recipe_id": recipe_ids,
            "line_index": indexes,
            "line_raw": raws,
            "quantity": quantities,
            "unit": units,
            "ingredient_text": texts,
            "normalized_text": normals,
        }
    )


def write_silver(lines: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    """Write both Silver tables to DuckDB, replacing them (idempotent re-run)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("parsed_lines", lines)
        con.execute(
            "CREATE OR REPLACE TABLE silver.recipe_ingredient_lines AS "
            "SELECT * FROM parsed_lines"
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE silver.distinct_ingredient_strings AS
            SELECT normalized_text,
                   COUNT(*) AS occurrence_count,
                   CASE WHEN COUNT(*) >= {HEAD_MIN} THEN 'head'
                        WHEN COUNT(*) >= {MID_MIN} THEN 'mid'
                        ELSE 'tail' END AS frequency_stratum
            FROM silver.recipe_ingredient_lines
            WHERE normalized_text <> ''
            GROUP BY normalized_text
            ORDER BY occurrence_count DESC
            """
        )
    finally:
        con.close()
    return db_path


def main() -> None:
    from pantryiq.lakehouse.catalog import get_catalog

    lines = build_line_rows(get_catalog())
    write_silver(lines)

    total = lines.num_rows
    with_qty = sum(q is not None for q in lines.column("quantity").to_pylist())
    with_unit = sum(u is not None for u in lines.column("unit").to_pylist())
    normals = lines.column("normalized_text").to_pylist()
    non_empty = sum(bool(n) for n in normals)
    print(f"silver.recipe_ingredient_lines: {total:,} lines")
    print(f"  with quantity : {with_qty:,} ({100 * with_qty / total:.1f}%)")
    print(f"  with unit     : {with_unit:,} ({100 * with_unit / total:.1f}%)")
    print(f"  non-empty text: {non_empty:,} ({100 * non_empty / total:.1f}%)")
    print(f"silver.distinct_ingredient_strings: {len(set(normals)):,} distinct")


if __name__ == "__main__":
    main()
