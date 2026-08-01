"""Distribution checks over published Gold — the part dbt tests express poorly.

§4 locks "dbt tests + Great Expectations", and for a while only the first existed. The honest
reason was that they overlap: dbt already covers bounds, uniqueness, nullability and referential
integrity, and adding a second tool to restate them would be ceremony.

**So this deliberately does not restate them.** The dbt gate asks "is any value impossible?" and
blocks on the answer. These expectations ask a different question — **"has the shape of the data
moved?"** — which bounds-checking cannot see:

- Every recipe could sit inside 0-910 kcal/100g and the *mean* still halve, because a conversion
  path broke. That is the pack-size bug (§14) in a form the gate passed cleanly.
- Coverage could hold at 63.9% per line while the *distribution* across recipes collapses.
- Row counts could stay identical while the tables are a stale vintage (`pipeline_run` exists to
  date the build; this checks it is actually recent).

**They are advisory, not a gate.** A drifted distribution is a finding to read, not a reason to
refuse to publish — the same split already used for `gram_accuracy`, which reports and does not
block. Expectations that fire on real data get muted, and a muted check protects nothing.

Ranges are set from the CURRENT published values with deliberate slack.

**Which statistic you pick matters more than the range.** I first asserted that these would have
caught the pack-size bug (§14 — a 3x mass overstatement on 218 recipes) and then checked, which
is the right order and was not the order I used. The median does **not** move at all: 652.0 g
before and after, because 218 of 15,000 recipes cannot shift a median. What moves is the tail —
p99 goes 4,172 g -> 12,517 g, a clean 3.00x. So the tail statistic is in the list and the claim
about the median is not.

Run:  uv run python -m pantryiq.gold.expectations   (needs `uv sync --group quality`)
"""
from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_DB = Path("data/pantryiq.duckdb")

# (label, SQL returning one number, low, high, why this range)
EXPECTATIONS = (
    (
        "mean recipe kcal/100g",
        "SELECT avg(kcal_per_100g) FROM gold.recipe_nutrition WHERE kcal_per_100g IS NOT NULL",
        150.0, 350.0,
        "a broken conversion path moves the CENTRE while every row stays inside 0-910",
    ),
    (
        "median recipe total grams",
        "SELECT median(total_grams) FROM gold.recipe_nutrition WHERE total_grams > 0",
        300.0, 1500.0,
        "the centre of the mass distribution; insensitive to tail bugs by design",
    ),
    (
        "p99 recipe total grams",
        "SELECT quantile_cont(total_grams, 0.99) FROM gold.recipe_nutrition WHERE total_grams > 0",
        2000.0, 7000.0,
        "VERIFIED to catch the pack-size bug: p99 moved 4,172 -> 12,517 g (3.00x) while the "
        "median did not move at all. A conversion-factor error lives in the tail",
    ),
    (
        "mean nutrition_coverage",
        "SELECT avg(nutrition_coverage) FROM gold.recipe_nutrition",
        0.45, 0.75,
        "coverage collapsing is the failure that most quietly degrades every answer",
    ),
    (
        "share of recipes with any nutrition",
        "SELECT count(*) FILTER (WHERE nutrition_coverage > 0)::double / count(*) "
        "FROM gold.recipe_nutrition",
        0.85, 1.0,
        "a resolution or portion regression shows here before it shows in the mean",
    ),
    (
        "mean data_trust_score",
        "SELECT avg(data_trust_score) FROM gold.recipe_nutrition",
        0.20, 0.55,
        "the score the product surfaces; drift means the thing users see has changed",
    ),
    (
        "recipes tagged, share",
        "SELECT count(DISTINCT recipe_id)::double / "
        "(SELECT count(*) FROM gold.recipe_nutrition) FROM gold.recipe_tags",
        0.02, 0.15,
        "tags fire only at full coverage, so this tracks completeness end to end",
    ),
    (
        "gram-conversion share of lines",
        "SELECT count(*) FILTER (WHERE grams IS NOT NULL)::double / count(*) "
        "FROM gold.recipe_ingredients_resolved",
        0.55, 0.72,
        "63.9% today; a swing either way means a conversion path changed behaviour",
    ),
    (
        "hours since publish",
        "SELECT date_diff('hour', max(published_at), now()) FROM gold.pipeline_run",
        0.0, 720.0,
        "a stale vintage passes every value check ever written — freshness needs its own",
    ),
)


def evaluate(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    """Run every expectation and return one result row each."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        results = []
        for label, query, low, high, why in EXPECTATIONS:
            value = con.execute(query).fetchone()[0]  # noqa: S608 — fixed literals above
            value = float(value) if value is not None else None
            results.append({
                "label": label, "value": value, "low": low, "high": high, "why": why,
                "ok": value is not None and low <= value <= high,
            })
    finally:
        con.close()
    return results


def main() -> None:
    results = evaluate()
    failures = [row for row in results if not row["ok"]]

    print("=" * 78)
    print("GOLD DISTRIBUTION EXPECTATIONS — advisory, not a gate")
    print("=" * 78)
    for row in results:
        value = "NULL" if row["value"] is None else f"{row['value']:.3f}"
        print(f"  {'ok  ' if row['ok'] else 'DRIFT'} {row['label']:34} {value:>9}  "
              f"expected [{row['low']}, {row['high']}]")
    print(f"\n  {len(results) - len(failures)}/{len(results)} within range")
    for row in failures:
        print(f"    {row['label']}: {row['why']}")
    print("\n  These do NOT block publication. The dbt gate enforces the impossible; these")
    print("  report movement, and a drifted distribution is a finding to read, not a refusal.")


if __name__ == "__main__":
    main()
