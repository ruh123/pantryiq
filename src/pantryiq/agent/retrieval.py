"""Structured retrieval over Gold — SQL, not vector search, and never under model control.

The brief is explicit (§6 Phase 4): rank candidates by ingredient overlap and constraint fit
using structured retrieval, not free-text similarity. Nothing here calls an LLM. The agent's
parse step turns a question into a `PantryQuery`; this module answers it from the warehouse;
generation then sees only what this returned. Retrieval being outside the model's control is
what makes "it can only speak from verified data" checkable rather than aspirational.

**Reads the exported Gold file**, not the shared warehouse. `gold/publish.py:export()` exists so
serving touches a file no writer holds a lock on — a reader on the shared file would block the
pipeline's next write, which is the wrong direction for a failure to travel.

**Matching is word-boundary with an optional plural, and that was measured rather than assumed.**
Three rules were compared over the real corpus:

    term      exact   substring   word-boundary   what the boundary kills
    corn         63       1,477             871   acorn, popcorn, mexicorn
    milk      2,306       3,842           3,436   buttermilk, milky way bar
    pepper    1,316       3,782           3,666   pepperoni, peppermint, cyapepper
    chicken     211       1,565           1,565   (nothing)

Exact match under-recalls catastrophically — `chicken` finds 211 recipes against 1,565 that
mention it. Substring over-recalls into different foods. The boundary rule keeps genuine plurals
(`eggs yolk`, `eggs separated`) while dropping `eggplant` and `popcorn`, and it generalizes: a
hand-curated stoplist would never have caught `cyapepper` or `mexicorn`.

**Latency is a design constraint, and the obvious SQL misses it by 20x.** Phase 4's budget is
<5 s end to end and effectively all of it belongs to the two Claude calls, so retrieval has 50 ms.
Expressing the match as a join against a table of patterns costs **973 ms** for three terms —
the pattern becomes a column, so the regex is recompiled per row across all 112,463 lines. The
same three terms bound as constants cost **17 ms**. See the comment in `retrieve`.

**The quality bar is a filter, not a ranking.** `min_coverage` defaults to 0.8 because a recipe
whose nutrition is 40% guessed cannot honestly answer a nutrition question. Only 4.7% of recipes
are fully covered, so requiring 1.0 would answer almost nothing — the coverage that *is* there
travels with every candidate so the answer can state it.

Run:  uv run python -m pantryiq.agent.retrieval
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

# The agent reads the published export, never the shared warehouse — see the docstring.
#
# `PANTRYIQ_GOLD_DB` overrides the location, because every path in this package is relative to the
# process CWD and a container's is not the repo root. Read at import time, matching how
# `profiles.yml` reads `PANTRYIQ_DB`: set it before the process starts, not during it. Callers that
# need a different file per call already pass `db_path` and are unaffected.
DEFAULT_DB = Path(os.environ.get("PANTRYIQ_GOLD_DB") or "data/pantryiq_gold.duckdb")

# A recipe below this is too incompletely weighed to answer a nutrition question about.
# 0.8 keeps 3,256 recipes; 1.0 would keep 711. See er_metrics.md §14 on why completeness is rare.
DEFAULT_MIN_COVERAGE = 0.8
DEFAULT_LIMIT = 8

# Pantry terms are user text interpolated into a regex. Anything outside this set is stripped so
# a query like "chicken|.*" cannot turn into a pattern that matches everything.
_SAFE_TERM = re.compile(r"[^a-z0-9 \-]")


@dataclass(frozen=True)
class PantryQuery:
    """What the user asked for, after parsing. Every field is optional but `pantry`."""

    pantry: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    max_kcal: float | None = None
    min_kcal: float | None = None
    # Which energy figure the kcal bounds apply to. "Under 500 calories" almost always means a
    # serving, and applying it to the total answers a meal question with sauces: the median
    # recipe here is 2,539 kcal total, so <=500 by total returns 145 recipes that are mostly
    # dips and dressings, while <=500 per serving returns 104 actual meals. Only 118 recipes
    # carry a per-serving figure (Phase 3 publishes it only at complete coverage), so "serving"
    # deliberately restricts the answer to the slice where that number is honest.
    kcal_basis: str = "total"  # "total" | "serving"
    tags: tuple[str, ...] = ()
    max_cost_usd: float | None = None
    min_coverage: float = DEFAULT_MIN_COVERAGE
    limit: int = DEFAULT_LIMIT


@dataclass(frozen=True)
class Candidate:
    """One retrieved recipe. Every field here is a fact from Gold, not a derivation."""

    recipe_id: str
    title: str | None
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    ingredient_count: int
    counted_ingredients: int
    total_kcal: float | None
    kcal_per_100g: float | None
    total_grams: float | None
    protein_g: float | None
    fat_g: float | None
    carb_g: float | None
    servings: int | None
    kcal_per_serving: float | None
    cost_total_usd: float | None
    cost_coverage: float | None
    nutrition_coverage: float
    data_trust_score: float
    tags: tuple[str, ...] = field(default=())


def term_pattern(term: str) -> str:
    """A word-boundary regex for one pantry term, tolerating a trailing plural.

    Sanitised first: the term is user-supplied and goes into a regex, so regex metacharacters
    would otherwise let a query widen its own match.
    """
    cleaned = _SAFE_TERM.sub("", term.strip().lower())
    return rf"\b{re.escape(cleaned)}s?\b" if cleaned else ""


def retrieve(query: PantryQuery, db_path: Path | str = DEFAULT_DB) -> list[Candidate]:
    """Recipes ranked by how many pantry terms they use, then by trust.

    Ranking by *count* of matched terms rather than a similarity score keeps the ordering
    explainable: a recipe is above another because it uses more of what you have, and the answer
    can say so.
    """
    patterns = [pattern for pattern in map(term_pattern, query.pantry) if pattern]
    exclusions = [pattern for pattern in map(term_pattern, query.exclude) if pattern]

    # Naming no ingredients and naming only ingredients we could not read are different requests,
    # and only the first may fall through to the trust ranking below. A term like ".*" sanitises
    # to nothing; treating that as "no pantry" would answer a question nobody asked with the whole
    # corpus, which reads as comprehension the agent does not have.
    if query.pantry and not patterns:
        return []

    # One CASE per pattern, each with the pattern **bound as a constant**, rather than a join
    # against a table of patterns. The join form reads better and is 57x slower: with the pattern
    # coming from a column, DuckDB recompiles the regex per row. Measured on the real export --
    # one regex over all 112,463 lines is 4.0 ms, but three via a join is 973 ms, well past this
    # module's 50 ms budget. As constants the same three cost 17 ms. Only the *number* of clauses
    # is assembled here; every pattern remains a bound parameter.
    matches = ", ".join(["CASE WHEN regexp_matches(normalized_text, ?) THEN ? END"] * len(patterns))
    # No pantry terms is a legitimate query ("something vegetarian under 500 kcal") -- every
    # recipe is a candidate, ranked by trust alone, rather than an empty answer.
    hits = (f"SELECT recipe_id, count(DISTINCT m) AS matched_count, list(DISTINCT m) AS "
            f"matched_patterns FROM (SELECT recipe_id, unnest([{matches}]) AS m "
            f"FROM gold.recipe_ingredients_resolved) WHERE m IS NOT NULL GROUP BY 1"
            if patterns else
            "SELECT recipe_id, 0 AS matched_count, []::VARCHAR[] AS matched_patterns "
            "FROM gold.recipe_nutrition")
    banned = " OR ".join(["regexp_matches(normalized_text, ?)"] * len(exclusions)) or "false"
    # Chosen from a closed set, never interpolated from user text.
    if query.kcal_basis not in ("total", "serving"):
        raise ValueError(f"kcal_basis must be 'total' or 'serving', got {query.kcal_basis!r}")
    kcal_column = "n.kcal_per_serving" if query.kcal_basis == "serving" else "n.total_kcal"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            f"""
            WITH hits AS ({hits}),
            excluded AS (
                SELECT DISTINCT recipe_id
                FROM gold.recipe_ingredients_resolved WHERE {banned}
            ),
            tagged AS (
                SELECT recipe_id, list(tag) AS tags
                FROM gold.recipe_tags GROUP BY 1
            )
            SELECT h.recipe_id, m.title, h.matched_patterns, h.matched_count,
                   n.ingredient_count, n.counted_ingredients,
                   n.total_kcal, n.kcal_per_100g, n.total_grams,
                   n.total_protein_g, n.total_fat_g, n.total_carb_g,
                   n.servings, n.kcal_per_serving,
                   n.cost_total_usd, n.cost_coverage,
                   n.nutrition_coverage, n.data_trust_score,
                   coalesce(g.tags, []) AS tags
            FROM hits h
            JOIN gold.recipe_nutrition n USING (recipe_id)
            LEFT JOIN gold.recipe_meta  m USING (recipe_id)
            LEFT JOIN tagged            g USING (recipe_id)
            WHERE h.recipe_id NOT IN (SELECT recipe_id FROM excluded)
              AND n.nutrition_coverage >= ?
              AND (? IS NULL OR {kcal_column} <= ?)
              AND (? IS NULL OR {kcal_column} >= ?)
              AND (? IS NULL OR n.cost_total_usd <= ?)
              -- Every requested tag must be present, not just one: "vegan and gluten-free"
              -- means both. An absent tag is unknown, never a denial (see recipe_tags.sql).
              AND len(list_intersect(coalesce(g.tags, []), ?::VARCHAR[])) = len(?::VARCHAR[])
            ORDER BY h.matched_count DESC, n.data_trust_score DESC, h.recipe_id
            LIMIT ?
            """,
            # Each pattern is bound twice: once to test, once as the label it yields.
            [value for pattern in patterns for value in (pattern, pattern)]
            + exclusions
            + [query.min_coverage,
               query.max_kcal, query.max_kcal, query.min_kcal, query.min_kcal,
               query.max_cost_usd, query.max_cost_usd,
               list(query.tags), list(query.tags),
               query.limit],
        ).fetchall()
    finally:
        con.close()

    # Map matched regex patterns back to the words the user actually typed — the answer should
    # say "chicken", not "\bchickens?\b".
    by_pattern = {term_pattern(term): term for term in query.pantry}
    candidates = []
    for row in rows:
        matched = tuple(sorted(by_pattern.get(p, p) for p in row[2]))
        candidates.append(Candidate(
            recipe_id=row[0], title=row[1], matched=matched,
            missing=tuple(sorted(set(query.pantry) - set(matched))),
            ingredient_count=row[4], counted_ingredients=row[5],
            total_kcal=row[6], kcal_per_100g=row[7], total_grams=row[8],
            protein_g=row[9], fat_g=row[10], carb_g=row[11],
            servings=row[12], kcal_per_serving=row[13],
            cost_total_usd=row[14], cost_coverage=row[15],
            nutrition_coverage=row[16], data_trust_score=row[17],
            tags=tuple(sorted(row[18])),
        ))
    return candidates


def main() -> None:
    import time

    demos = (
        PantryQuery(pantry=("chicken", "rice", "onion")),
        PantryQuery(pantry=("egg", "flour", "sugar"), max_kcal=2500),
        PantryQuery(pantry=("tomato", "garlic"), tags=("vegetarian",)),
        PantryQuery(pantry=("beef",), exclude=("onion",), max_cost_usd=8.0),
    )
    for query in demos:
        started = time.perf_counter()
        results = retrieve(query)
        elapsed = (time.perf_counter() - started) * 1000

        wanted = " + ".join(query.pantry)
        limits = ", ".join(filter(None, [
            f"<={query.max_kcal:.0f} kcal" if query.max_kcal else "",
            f"<=${query.max_cost_usd:.2f}" if query.max_cost_usd else "",
            f"tags={','.join(query.tags)}" if query.tags else "",
            f"no {','.join(query.exclude)}" if query.exclude else "",
        ]))
        print(f"\n{wanted}{'  (' + limits + ')' if limits else ''}   "
              f"{len(results)} hits in {elapsed:.1f} ms")
        for candidate in results[:4]:
            kcal = f"{candidate.total_kcal:,.0f} kcal" if candidate.total_kcal else "no kcal"
            print(f"  {str(candidate.title)[:38]:40} {kcal:>12}  "
                  f"cov {candidate.nutrition_coverage:.0%}  trust {candidate.data_trust_score:.2f}  "
                  f"has {'+'.join(candidate.matched)}")

    print("\n  Ranking is by COUNT of pantry terms matched, then trust — so an answer can always")
    print("  say why one recipe outranked another. Nothing here consults a model.")


if __name__ == "__main__":
    main()
