"""2.7 — `silver.ingredient_entity_map`, and the headline numbers.

The deliverable the measurement sections build toward: every one of the 9,324 distinct ingredient
strings resolved to a USDA entity with its nutrition attached, plus a confidence flag, plus the
occurrence count that says how much each string actually matters.

The headline is a **stratified estimate**, not an average over the gold set. The 300 gold labels
were drawn 100/100/100 from strata holding 545 / 1,580 / 7,228 distinct strings, so the tail is
over-sampled by roughly 70× relative to the head. Averaging the sample directly would answer "how
does the pipeline do on the gold set", which nobody is asking. Reweighting by stratum size answers
"how does it do on the corpus", and reweighting by *occurrences within* each stratum answers "how
does it do on the ingredient lines a user actually hits" — the brief asks for both denominators
and they differ substantially, because head strings carry 83.5% of all occurrences.

Two limits stated up front, because the headline is meaningless without them:

- **The estimate inherits the gold labels' quality.** §9 measured Krippendorff's α = 0.709 across
  three independent annotators. These numbers are as good as those labels and no better.
- **The resolver cannot say `no-match`.** It always returns its best candidate, so the 12% of
  strings that genuinely have no USDA entity are silently assigned one. The confidence flag is
  the only proxy, and its correlation with the true null class is reported rather than assumed.

Run:  uv run python -m pantryiq.er.entity_map
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from pantryiq.er.evaluate import ITERATIONS, SEED
from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels
from pantryiq.er.nutrition import EQUIVALENT, error
from pantryiq.er.relabel import same_entity
from pantryiq.er.resolve import LOW_CONFIDENCE, resolve

STRATA = ("head", "mid", "tail")


def build(db_path: Path | str = DEFAULT_DB) -> pa.Table:
    """One row per distinct ingredient string: its entity, nutrition, confidence and weight."""
    picks = {text: (fdc_id, cosine, confidence, flagged)
             for text, fdc_id, cosine, confidence, flagged in resolve(db_path)}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        strings = con.execute(
            "SELECT normalized_text, occurrence_count, frequency_stratum "
            "FROM silver.distinct_ingredient_strings ORDER BY normalized_text"
        ).fetchall()
        foods = {row[0]: row[1:] for row in con.execute(
            "SELECT fdc_id, description_raw, kcal_per_100g, protein_g, fat_g, carb_g, "
            "is_deprioritized FROM silver.usda_foods").fetchall()}
    finally:
        con.close()

    columns: dict[str, list] = {key: [] for key in (
        "normalized_text", "occurrence_count", "frequency_stratum", "fdc_id", "description_raw",
        "kcal_per_100g", "protein_g", "fat_g", "carb_g", "is_deprioritized", "cosine",
        "confidence", "flagged")}
    for text, occurrences, stratum in strings:
        if text not in picks:
            continue  # no candidates were generated for this string
        fdc_id, cosine, confidence, flagged = picks[text]
        description, kcal, protein, fat, carb, deprioritized = foods[fdc_id]
        for key, value in (
            ("normalized_text", text), ("occurrence_count", occurrences),
            ("frequency_stratum", stratum), ("fdc_id", fdc_id),
            ("description_raw", description), ("kcal_per_100g", kcal), ("protein_g", protein),
            ("fat_g", fat), ("carb_g", carb), ("is_deprioritized", deprioritized),
            ("cosine", cosine), ("confidence", confidence), ("flagged", flagged),
        ):
            columns[key].append(value)
    return pa.table(columns)


def write(table: pa.Table, db_path: Path | str = DEFAULT_DB) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("entity_map_rows", table)
        con.execute("CREATE OR REPLACE TABLE silver.ingredient_entity_map AS "
                    "SELECT * FROM entity_map_rows")
    finally:
        con.close()


def corpus_totals(db_path: Path | str = DEFAULT_DB) -> dict[str, tuple[int, int]]:
    """{stratum: (distinct strings, total occurrences)} over the whole corpus, not the sample."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT frequency_stratum, count(*), sum(occurrence_count) "
            "FROM silver.distinct_ingredient_strings GROUP BY 1"
        ).fetchall()
    finally:
        con.close()
    return {stratum: (int(count), int(occurrences)) for stratum, count, occurrences in rows}


def stratified_rate(by_stratum: dict[str, list[tuple[float, bool]]],
                    totals: dict[str, tuple[int, int]], weighted: bool) -> float:
    """Reweight per-stratum sample rates by the stratum's true size in the corpus.

    `weighted=False` sizes each stratum by its distinct-string count and treats sampled strings
    equally — the per-unique-string denominator. `weighted=True` sizes by total occurrences and
    weights each sampled string by its own occurrence count — the per-occurrence denominator,
    which is what a user of the product actually experiences.
    """
    numerator = denominator = 0.0
    for stratum, rows in by_stratum.items():
        if not rows or stratum not in totals:
            continue
        distinct, occurrences = totals[stratum]
        size = occurrences if weighted else distinct
        if weighted:
            total_weight = sum(weight for weight, _ in rows)
            if not total_weight:
                continue
            rate = sum(weight for weight, hit in rows if hit) / total_weight
        else:
            rate = sum(1 for _, hit in rows if hit) / len(rows)
        numerator += size * rate
        denominator += size
    return numerator / denominator if denominator else 0.0


def stratified_ci(by_stratum: dict[str, list[tuple[float, bool]]],
                  totals: dict[str, tuple[int, int]], weighted: bool,
                  iterations: int = ITERATIONS, seed: int = SEED) -> tuple[float, float, float]:
    """(estimate, CI low, CI high) — resampling *within* each stratum, which is how the sample
    was drawn. Pooling the strata before resampling would understate the interval."""
    estimate = stratified_rate(by_stratum, totals, weighted)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        resampled = {}
        for stratum, rows in by_stratum.items():
            if rows:
                index = rng.integers(0, len(rows), size=len(rows))
                resampled[stratum] = [rows[position] for position in index]
        draws.append(stratified_rate(resampled, totals, weighted))
    low, high = np.percentile(draws, [2.5, 97.5])
    return (estimate, float(low), float(high))


def scored_sample(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT):
    """Per gold-labeled resolvable string: (stratum, occurrences, entity hit, kcal-equivalent)."""
    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        weights = dict(con.execute(
            "SELECT normalized_text, occurrence_count FROM silver.distinct_ingredient_strings"
        ).fetchall())
        strata = dict(con.execute(
            "SELECT normalized_text, frequency_stratum FROM silver.distinct_ingredient_strings"
        ).fetchall())
        usda = con.execute(
            "SELECT fdc_id, description_raw, kcal_per_100g FROM silver.usda_foods").fetchall()
        picks = dict(con.execute(
            "SELECT normalized_text, fdc_id FROM silver.ingredient_entity_map").fetchall())
    finally:
        con.close()
    description = {row[0]: row[1] for row in usda}
    kcal = {row[0]: row[2] for row in usda}

    rows = []
    for text, record in labels.items():
        gold = record["fdc_id"]
        if gold == NO_MATCH or text not in picks or text not in strata:
            continue
        pick = picks[text]
        gap = error(kcal.get(gold), kcal.get(pick))
        rows.append({
            "stratum": strata[text],
            "weight": float(weights.get(text, 1)),
            "entity": same_entity(pick, gold, description),
            "equivalent": None if gap is None else gap < EQUIVALENT,
        })
    return rows


def main(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT) -> None:
    table = build(db_path)
    write(table, db_path)
    occurrences = table.column("occurrence_count").to_pylist()
    kcal = table.column("kcal_per_100g").to_pylist()
    flagged = table.column("flagged").to_pylist()
    total_occurrences = sum(occurrences)

    print("=" * 78)
    print("2.7 — silver.ingredient_entity_map")
    print("=" * 78)
    print(f"  {table.num_rows:,} distinct ingredient strings resolved, covering "
          f"{total_occurrences:,} ingredient-line occurrences")
    with_kcal = sum(1 for value in kcal if value is not None)
    kcal_occurrences = sum(o for o, value in zip(occurrences, kcal) if value is not None)
    print(f"  with an energy value: {with_kcal:,} strings ({100 * with_kcal / table.num_rows:.1f}%)"
          f"  |  {kcal_occurrences:,} occurrences "
          f"({100 * kcal_occurrences / total_occurrences:.1f}%)")
    flagged_count = sum(flagged)
    flagged_occurrences = sum(o for o, value in zip(occurrences, flagged) if value)
    print(f"  flagged low-confidence:  {flagged_count:,} strings "
          f"({100 * flagged_count / table.num_rows:.1f}%)"
          f"  |  {flagged_occurrences:,} occurrences "
          f"({100 * flagged_occurrences / total_occurrences:.1f}%)")
    print(f"  -> flagged strings are {100 * flagged_count / table.num_rows:.1f}% of the "
          f"vocabulary but only {100 * flagged_occurrences / total_occurrences:.1f}% of what a "
          "user hits:\n     the hard strings are mostly rare ones.")

    totals = corpus_totals(db_path)
    print("\n  corpus strata")
    for stratum in STRATA:
        distinct, occurrence_total = totals[stratum]
        print(f"    {stratum:5} {distinct:6,} distinct strings "
              f"({100 * distinct / sum(t[0] for t in totals.values()):5.1f}%)   "
              f"{occurrence_total:8,} occurrences "
              f"({100 * occurrence_total / sum(t[1] for t in totals.values()):5.1f}%)")

    # --- the headline ------------------------------------------------------------------
    sample = scored_sample(db_path, gold_dir)
    print("\n" + "=" * 78)
    print("HEADLINE — stratified estimate over the corpus")
    print("=" * 78)
    print(f"  from {len(sample)} gold-labeled resolvable strings, reweighted by true stratum size")
    for label, key in (("nutritionally equivalent (<10% kcal)", "equivalent"),
                       ("entity top-1", "entity")):
        by_stratum = {stratum: [(row["weight"], bool(row[key])) for row in sample
                                if row["stratum"] == stratum and row[key] is not None]
                      for stratum in STRATA}
        per_string = stratified_ci(by_stratum, totals, weighted=False)
        per_occurrence = stratified_ci(by_stratum, totals, weighted=True)
        print(f"\n  {label}")
        print(f"    per unique string : {per_string[0]:6.1%}  "
              f"[{per_string[1]:.1%}, {per_string[2]:.1%}]")
        print(f"    per occurrence    : {per_occurrence[0]:6.1%}  "
              f"[{per_occurrence[1]:.1%}, {per_occurrence[2]:.1%}]   <- what a user experiences")
        for stratum in STRATA:
            rows = by_stratum[stratum]
            if rows:
                print(f"      {stratum:5} n={len(rows):3}  "
                      f"{sum(1 for _, hit in rows if hit) / len(rows):6.1%}")

    print("\n  Both figures inherit the gold labels' quality: Krippendorff's alpha = 0.709 (§9).")
    print("  They are estimates of agreement with those labels, not with ground truth.")

    # --- the null-class limitation ------------------------------------------------------
    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    flag_of = dict(zip(table.column("normalized_text").to_pylist(), flagged))
    null_strings = [text for text, record in labels.items()
                    if record["fdc_id"] == NO_MATCH and text in flag_of]
    real_strings = [text for text, record in labels.items()
                    if record["fdc_id"] != NO_MATCH and text in flag_of]
    if null_strings and real_strings:
        print("\n" + "=" * 78)
        print("LIMITATION — the resolver cannot say no-match")
        print("=" * 78)
        print("  It always returns its best candidate, so a string with no real USDA entity still")
        print("  gets one. The confidence flag is the only proxy available:")
        print(f"    gold no-match  ({len(null_strings):3} strings): "
              f"{100 * sum(flag_of[t] for t in null_strings) / len(null_strings):5.1f}% flagged")
        print(f"    gold resolvable({len(real_strings):3} strings): "
              f"{100 * sum(flag_of[t] for t in real_strings) / len(real_strings):5.1f}% flagged")
        print(f"  A flag threshold of {LOW_CONFIDENCE} was fitted for nutrition error, not for "
              "null detection.")


if __name__ == "__main__":
    main()
