"""Nutrition error — the metric the product is actually judged on.

Entity-level precision asks "did you pick entity 172194 or 170878?", and two careful labelers
following the same guide disagree on that often enough to make the number partly a measure of
convention rather than correctness (see `docs/er_metrics.md`). But nothing downstream consumes
an `fdc_id`; it consumes kcal per 100g. So the honest question is not "was the entity right"
but "how wrong is the nutrition".

That reframing is not a way of making disagreements disappear — measured against pass 1 it
confirms 9 of the 13 comparable ones were real. What it does is separate the disputes that matter
from the ones that never did: `spiral pasta` (enriched vs unenriched) is a **0.0%** kcal
difference, `rounded tbsp flour` 0.5%, and an entity-identity metric charges each as a full
error.

Two numbers this module produces:

- **error** — the relative kcal gap between a predicted entity and the gold one. Denominated by
  the *larger* of the two, so it is symmetric and bounded at 100%: relative-to-gold explodes on
  near-zero-kcal foods (a 32 vs 254 kcal gap reads as 694%, which tells you nothing useful about
  a food that is mostly water).
- **ambiguity** — the kcal spread across entities for the *same food* as the gold label. This is
  the error a perfect entity resolver still cannot avoid, because the ingredient line does not
  say which facet it means. It needs no labels and it bounds what any precision claim can mean.

  Read it as an **upper** bound. "Same food" is the two leading facets, which is right for
  `('beans', 'pinto')` — dry 333 vs canned 82 kcal is a genuine ambiguity the line leaves open
  — but too coarse for `('beverages', 'tea')`, where it groups 33 unrelated teas. Group size is
  reported alongside so the over-grouped cases are visible rather than buried.

Run:  uv run python -m pantryiq.er.nutrition
"""
from __future__ import annotations

import statistics
from pathlib import Path

import duckdb

from pantryiq.er.convention import candidates_by_string, food_key
from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels
from pantryiq.er.relabel import paths, wilson

# Bands, not a single mean: a 3% gap and a 60% gap are different kinds of event, and averaging
# them hides which one the pipeline actually makes.
BANDS = ((0.10, "within 10%  (nutritionally equivalent)"),
         (0.25, "10-25%"),
         (0.50, "25-50%"),
         (1.01, "over 50%    (materially wrong)"))
EQUIVALENT = 0.10
# Below this absolute gap, a relative error is not meaningful. Without it the bounded ratio is
# degenerate whenever the smaller value is near zero: `Beverages, tea` spans 0 to 1 kcal/100g,
# and |0-1|/1 reports 100% — filing brewed tea under "materially wrong".
MATERIAL_KCAL = 5.0


def error(first: float | None, second: float | None,
          floor: float = MATERIAL_KCAL) -> float | None:
    """Relative kcal gap, symmetric and bounded at 1.0. None when either value is missing.

    73 USDA foods carry no energy value at all, so a missing kcal is a real case and must not
    silently become a zero error.
    """
    if first is None or second is None:
        return None
    largest = max(first, second)
    if largest <= 0 or abs(first - second) < floor:
        return 0.0
    return abs(first - second) / largest


def band_of(value: float) -> str:
    return next(label for threshold, label in BANDS if value < threshold)


def distribution(errors: list[float]) -> dict[str, int]:
    counts = {label: 0 for _, label in BANDS}
    for value in errors:
        counts[band_of(value)] += 1
    return counts


def ambiguity(gold_fdc: str, candidates: list[tuple], description: dict[str, str],
              kcal: dict[str, float | None]) -> float | None:
    """Widest relative kcal gap among candidate entities naming the same food as the gold label.

    This is the irreducible part: if the line is `milk` and USDA offers whole, 2%, 1% and skim,
    a resolver that identifies the food perfectly can still be this far off on nutrition, and no
    amount of labeling removes it.
    """
    gold_description = description.get(gold_fdc)
    if gold_description is None:
        return None
    key = food_key(gold_description)
    values = [kcal[row[0]] for row in candidates
              if food_key(row[1]) == key and kcal.get(row[0]) is not None]
    if len(values) < 2:
        return None
    return error(min(values), max(values))


def _load(db_path: Path | str, strings: list[str]):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT fdc_id, description_raw, kcal_per_100g FROM silver.usda_foods").fetchall()
        return ({r[0]: r[1] for r in rows}, {r[0]: r[2] for r in rows},
                candidates_by_string(con, strings))
    finally:
        con.close()


def _print_distribution(errors: list[float], indent: str = "  ") -> None:
    counts = distribution(errors)
    for _, label in BANDS:
        print(f"{indent}{label:44} {counts[label]:3}/{len(errors)}")
    print(f"{indent}median {100 * statistics.median(errors):.1f}%   "
          f"mean {100 * statistics.mean(errors):.1f}%")


def main(out_dir: Path | str = DEFAULT_OUT, db_path: Path | str = DEFAULT_DB) -> None:
    labels = load_labels(Path(out_dir) / "labels.jsonl")
    resolved = {text: record["fdc_id"] for text, record in labels.items()
                if record["fdc_id"] != NO_MATCH}
    description, kcal, candidates = _load(db_path, list(resolved))

    print("IRREDUCIBLE AMBIGUITY — kcal spread among entities for the same food as the gold "
          "label.\nNo labels needed; this bounds what any precision claim can mean.\n")
    bounds, sizes = [], []
    for text, gold in sorted(resolved.items()):
        value = ambiguity(gold, candidates.get(text, []), description, kcal)
        if value is None:
            continue
        bounds.append(value)
        key = food_key(description[gold])
        sizes.append((sum(1 for row in candidates.get(text, []) if food_key(row[1]) == key),
                      text))
    print(f"  measurable on {len(bounds)} of {len(resolved)} resolved labels")
    _print_distribution(bounds)
    wide = [(size, text) for size, text in sizes if size >= 20]
    print(f"  over-grouped (>=20 same-key entities, treat as inflated): {len(wide)} of "
          f"{len(bounds)}" + (f" — e.g. {sorted(wide, reverse=True)[0][1]!r}" if wide else ""))

    _, judgments_path = paths(out_dir, 1)
    relabeled = load_labels(judgments_path)
    if not relabeled:
        return

    print("\nPASS 1 RELABEL, IN KCAL TERMS — the same disagreements, priced.\n")
    errors, null_disagreements, missing = [], [], []
    for text, record in sorted(relabeled.items()):
        gold = resolved.get(text, NO_MATCH if text in labels else None)
        if gold is None:
            continue
        theirs = record["fdc_id"]
        if gold == NO_MATCH or theirs == NO_MATCH:
            if gold != theirs:
                null_disagreements.append(text)
            continue
        value = error(kcal.get(gold), kcal.get(theirs))
        (errors if value is not None else missing).append(value if value is not None else text)

    print(f"  comparable pairs: {len(errors)}   "
          f"null-class disagreements: {len(null_disagreements)}   no kcal: {len(missing)}")
    _print_distribution(errors)
    equivalent = sum(1 for value in errors if value < EQUIVALENT)
    low, high = wilson(equivalent, len(errors))
    print(f"\n  nutritionally equivalent (<{100 * EQUIVALENT:.0f}%): {equivalent}/{len(errors)} "
          f"= {100 * equivalent / len(errors):.1f}% (95% Wilson CI "
          f"{100 * low:.1f}-{100 * high:.1f}%)")
    print("  — compare 36.0% strict entity agreement on the same pass: the gap is disputes "
          "that\n    cost nothing nutritionally, which an entity metric charges as full errors.")


if __name__ == "__main__":
    main()
