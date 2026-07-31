"""Measure whether the gram conversions are RIGHT, not just whether they exist (3.3b).

3.3 reports coverage — 58.9% of lines converted. Coverage says nothing about accuracy, and
every Gold number is `grams x kcal_per_100g / 100`, so an error here multiplies straight into
recipe nutrition. Phase 2 measured the second factor exhaustively (300 gold labels, §1-§13) and
nothing measured the first. This closes that asymmetry.

**The ground truth here is external, which makes it stronger than the ER labels.** "Which USDA
entity does 'evaporated milk' mean" is a judgment call, and the §9 ensemble put those labels at
alpha = 0.721. "How much does a cup of granulated sugar weigh" has a published answer — 200 g —
that a reader can check. `data/gram_reference.csv` carries the value and its source for the
highest-occurrence (ingredient, unit) pairs; the 34 seeded pairs cover ~46% of all converted
lines, so this is a head-weighted check by design.

**It measures the pipeline end to end, not just the arithmetic.** A conversion is wrong if the
gram weight is wrong OR if the resolver picked the wrong food and its density came along —
which is how the corpus's single most common conversion (`sugar` + `cup`, 4,170 lines) lands at
110 g against a true 200 g: bare "sugar" resolves to *powdered* sugar. §11's kcal metric could
not see that; in mass terms it is a 45% error.

Run:  uv run python -m pantryiq.silver.gram_accuracy
"""
from __future__ import annotations

import csv
from pathlib import Path

import duckdb

from pantryiq.er.relabel import wilson

DEFAULT_DB = Path("data/pantryiq.duckdb")
REFERENCE_PATH = Path("data/gram_reference.csv")

# A conversion counts as correct within 10% of the reference, matching the equivalence band the
# project already uses for nutrition (nutrition.EQUIVALENT). Cooking references themselves
# disagree by a few percent — flour is quoted at 120-125 g/cup — so a tighter band would measure
# the disagreement between sources rather than the pipeline.
TOLERANCE = 0.10


def load_reference(path: Path | str = REFERENCE_PATH) -> dict[tuple[str, str], float]:
    """{(normalized_text, unit): grams per one unit} from the curated reference table."""
    with Path(path).open(encoding="utf-8") as handle:
        return {(row["normalized_text"], row["unit"]): float(row["grams_per_unit"])
                for row in csv.DictReader(handle)}


def observed(db_path: Path | str = DEFAULT_DB) -> dict[tuple[str, str], tuple[float, str, int]]:
    """{(text, unit): (grams per unit, method, occurrences)} as the pipeline actually computes."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT l.normalized_text, l.unit,
                   median(g.grams / l.quantity) AS grams_per_unit,
                   min(g.method)                AS method,
                   count(*)                     AS occurrences
            FROM silver.recipe_ingredient_lines l
            JOIN silver.recipe_ingredient_grams g
              ON g.recipe_id = l.recipe_id AND g.line_index = l.line_index
            WHERE g.converted AND l.unit IS NOT NULL AND l.quantity > 0
            GROUP BY 1, 2
            """
        ).fetchall()
    finally:
        con.close()
    return {(text, unit): (grams, method, count) for text, unit, grams, method, count in rows}


def relative_error(actual: float, expected: float) -> float:
    """Symmetric relative gap, denominated by the larger value so it is bounded at 1.0.

    Same shape as `nutrition.error`, deliberately: the two metrics are read side by side and a
    reader should not have to hold two definitions of "off by X%" at once.
    """
    largest = max(actual, expected)
    return 0.0 if largest <= 0 else abs(actual - expected) / largest


def compare(reference: dict, seen: dict) -> list[dict]:
    """One row per reference pair the pipeline actually converted."""
    rows = []
    for (text, unit), expected in sorted(reference.items()):
        if (text, unit) not in seen:
            continue
        actual, method, occurrences = seen[(text, unit)]
        error = relative_error(actual, expected)
        rows.append({
            "text": text, "unit": unit, "expected": expected, "actual": actual,
            "method": method, "occurrences": occurrences,
            "error": error, "correct": error < TOLERANCE,
        })
    return rows


def main(db_path: Path | str = DEFAULT_DB, reference_path: Path | str = REFERENCE_PATH) -> None:
    rows = compare(load_reference(reference_path), observed(db_path))
    if not rows:
        print("no reference pairs matched — has the pipeline been run?")
        return

    total_occurrences = sum(row["occurrences"] for row in rows)
    correct = sum(1 for row in rows if row["correct"])
    weighted = sum(row["occurrences"] for row in rows if row["correct"])
    low, high = wilson(correct, len(rows))

    print("=" * 78)
    print("3.3b — GRAM CONVERSION ACCURACY (not coverage)")
    print("=" * 78)
    print(f"  {len(rows)} reference pairs, covering {total_occurrences:,} converted lines")
    print(f"  within {TOLERANCE:.0%} of the reference:")
    print(f"    per pair       : {correct}/{len(rows)} = {100 * correct / len(rows):.1f}%  "
          f"[{100 * low:.1f}%, {100 * high:.1f}%]")
    print(f"    per occurrence : {100 * weighted / total_occurrences:.1f}%  "
          "<- what a user actually hits")

    print("\n  by conversion path")
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        hits = sum(1 for row in subset if row["correct"])
        occ = sum(row["occurrences"] for row in subset)
        occ_hits = sum(row["occurrences"] for row in subset if row["correct"])
        print(f"    {method:20} {hits:2}/{len(subset):2} pairs   "
              f"{100 * occ_hits / occ:5.1f}% of {occ:,} lines")

    wrong = sorted((row for row in rows if not row["correct"]),
                   key=lambda row: -row["occurrences"])
    if wrong:
        print(f"\n  WRONG — {sum(row['occurrences'] for row in wrong):,} lines "
              f"({100 * sum(row['occurrences'] for row in wrong) / total_occurrences:.1f}% "
              "of those checked)")
        print(f"    {'ingredient':22} {'unit':11} {'got':>8} {'expected':>9} {'error':>7}  lines")
        for row in wrong:
            print(f"    {row['text'][:20]:22} {row['unit']:11} {row['actual']:8.1f} "
                  f"{row['expected']:9.1f} {row['error']:6.1%}  {row['occurrences']:,}")

    print("\n  Reference values are published figures with their source recorded in")
    print("  data/gram_reference.csv — unlike the ER labels, a reader can check them.")
    print("  A wrong row is the WHOLE pipeline being wrong: bad gram weight, or the resolver")
    print("  picking the wrong food and its density coming with it.")


if __name__ == "__main__":
    main()
