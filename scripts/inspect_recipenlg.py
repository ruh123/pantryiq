"""Read-only RecipeNLG format check — samples the first N rows, never loads the full ~2GB file.

Phase 1 first task (see docs/PantryIQ_Master_Prompt.md §6): confirm that RecipeNLG's
`ingredients` are full raw lines carrying quantities, and that `NER` gives clean names.

Usage:
    python scripts/inspect_recipenlg.py [path/to/full_dataset.csv] [n_rows]
"""
import ast
import csv
import json
import re
import sys

DEFAULT_PATH = "data/raw/recipenlg/full_dataset.csv"
QTY = re.compile(r"^\s*(\d+\s*\d*/?\d*|\d+\.\d+|[¼½¾⅓⅔⅕⅖⅗⅘⅛⅜⅝⅞])")

csv.field_size_limit(10_000_000)


def parse_list(cell):
    """Parse a stringified list cell (JSON first, then Python literal)."""
    for fn in (json.loads, ast.literal_eval):
        try:
            value = fn(cell)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list):
            return value
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    rows = ing_lines = ing_with_qty = ner_total = bad_ing = bad_ner = 0
    examples = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows += 1
            ings = parse_list(row[2])
            ners = parse_list(row[6])
            if ings is None:
                bad_ing += 1
                continue
            if ners is None:
                bad_ner += 1
            ing_lines += len(ings)
            ing_with_qty += sum(1 for x in ings if QTY.match(x))
            ner_total += len(ners) if ners else 0
            if len(examples) < 4:
                examples.append((row[1][:60], ings[:3], (ners or [])[:3]))

    pct = 100 * ing_with_qty / max(ing_lines, 1)
    print(f"header cols: {header}")
    print(f"sampled recipes: {rows}  (unparseable ingredients: {bad_ing}, NER: {bad_ner})")
    print(f"total ingredient lines: {ing_lines}  avg/recipe: {ing_lines / max(rows, 1):.1f}")
    print(f"lines with a leading quantity: {ing_with_qty}/{ing_lines} = {pct:.1f}%")
    print(f"NER names: {ner_total}  avg/recipe: {ner_total / max(rows, 1):.1f}")
    print("\n--- examples (title | first 3 ingredient lines | first 3 NER) ---")
    for title, ings, ners in examples:
        print(f"\nTITLE: {title}")
        print(f"  ingredients: {ings}")
        print(f"  NER        : {ners}")


if __name__ == "__main__":
    main()
