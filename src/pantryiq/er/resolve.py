"""The shipped resolver: top embedding cosine, plus a confidence flag.

This is what runs. It is deliberately the simplest thing that works, because 2.5 measured the
alternative and the alternative earned nothing: eight engineered features, a logistic regression
and isotonic calibration came out +3.8pp on the tune split and −3.9pp on the holdout — the
direction flipped, which is what a noise difference does. See `docs/er_metrics.md` §7.

`scoring.py`, `features.py` and `routing.py` are kept as the reproducible record of that
experiment. They are **not** in this path. Nothing here imports them.

What ships:

- **pick** — the candidate with the highest `cosine_search`, which is `rank = 0` by construction.
- **confidence** — that cosine mapped through an isotonic curve fitted on the tune split, so the
  number means "estimated probability this pick is within 10% of the right calories". 2.5 showed
  a confidence signal cannot certify correctness (there is no subset that is reliably right), but
  it does identify likely failures, and that is what the flag is for.

The confidence curve is fitted on tune only and stored, so resolving is a lookup, not a fit.

Run:  uv run python -m pantryiq.er.resolve
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb
import numpy as np

from pantryiq.er.abstain import load_threshold
from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels, load_sample
from pantryiq.er.nutrition import EQUIVALENT, error

CURVE_PATH = Path("data/confidence_curve.json")
OVERRIDES_PATH = Path("data/entity_overrides.csv")
# Chosen on the tune split, not the holdout. The isotonic curve is a coarse step function, so
# cuts cluster: anything in 0.40-0.50 flags the same 23% of tune strings, and that band is 35%
# nutritionally equivalent against 63% for the rest. A lower cut (0.35) flagged 2 strings out of
# 173 — technically 0% equivalent, and useless as an alarm.
LOW_CONFIDENCE = 0.40


def top_picks(db_path: Path | str = DEFAULT_DB, strings: list[str] | None = None):
    """[(normalized_text, fdc_id, cosine)] — the best candidate per string, by cosine.

    Ties break on the lowest `fdc_id`, and the outer `ORDER BY` is not decoration: without it
    DuckDB returns the same picks in a different row order on successive identical calls, which
    makes run-to-run diffs of the resolved output noise.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        selection = ("" if strings is None
                     else " AND normalized_text IN (SELECT UNNEST(?))")
        query = f"""
            SELECT normalized_text, fdc_id, cosine_search
            FROM (SELECT *, row_number() OVER (PARTITION BY normalized_text
                                               ORDER BY cosine_search DESC, fdc_id) AS position
                  FROM silver.ingredient_candidates)
            WHERE position = 1{selection}
            ORDER BY normalized_text
        """  # noqa: S608 — `selection` is a fixed literal, the value is bound
        parameters = [] if strings is None else [strings]
        return con.execute(query, parameters).fetchall()
    finally:
        con.close()


def fit_curve(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT):
    """Isotonic map from cosine to P(within 10% of the right calories), fitted on tune only."""
    from sklearn.isotonic import IsotonicRegression

    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    tune = {row["normalized_text"] for row in load_sample(gold_dir) if row["split"] == "tune"}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        kcal = dict(con.execute(
            "SELECT fdc_id, kcal_per_100g FROM silver.usda_foods").fetchall())
    finally:
        con.close()

    cosines, outcomes = [], []
    for text, fdc_id, cosine in top_picks(db_path, sorted(tune)):
        truth = labels[text]["fdc_id"]
        if truth == NO_MATCH:
            continue
        gap = error(kcal.get(truth), kcal.get(fdc_id))
        if gap is not None:
            cosines.append(cosine)
            outcomes.append(float(gap < EQUIVALENT))
    curve = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cosines, outcomes)
    return curve, cosines, outcomes


def save_curve(curve, path: Path | str = CURVE_PATH) -> Path:
    """Store the fitted curve as plain thresholds, so resolving needs no training step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "x": [float(value) for value in curve.X_thresholds_],
        "y": [float(value) for value in curve.y_thresholds_],
        "equivalent_within": EQUIVALENT,
        "fitted_on": "tune split (201 strings)",
    }, indent=2) + "\n", encoding="utf-8")
    return path


def load_curve(path: Path | str = CURVE_PATH):
    """Return a callable cosine -> confidence, interpolating the stored curve."""
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    xs, ys = np.array(stored["x"]), np.array(stored["y"])

    def confidence(cosine):
        return float(np.interp(cosine, xs, ys))

    return confidence


def load_overrides(path: Path | str = OVERRIDES_PATH) -> dict[str, str]:
    """{normalized_text: fdc_id} — hand-corrected picks for head strings.

    These are ASSERTIONS, not learned, and they exist because a measured, principled re-ranking
    rule could not fix them (see er_metrics.md §14): embedding similarity puts `egg` on Eggnog,
    `milk` on Crackers-milk and `sugar` on powdered sugar, and the kcal-based headline is blind
    to the last one because powdered and granulated sugar carry near-identical energy.

    Deliberately bounded and auditable: only strings listed in the CSV change, each row records
    what it replaces and why, and every one can be checked against USDA by a reader. Returns {}
    when the file is absent, so the resolver behaves exactly as before without it.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["normalized_text"]: row["fdc_id"] for row in csv.DictReader(handle)}


def resolve(db_path: Path | str = DEFAULT_DB, curve_path: Path | str = CURVE_PATH,
            strings: list[str] | None = None, abstain_threshold: float | None = None):
    """[(text, fdc_id, cosine, confidence, flagged, abstained)] for every distinct string.

    Two independent signals, because they answer different questions:

    - `flagged` — the isotonic confidence curve, fitted for *nutrition error*. "This pick may
      have the wrong facet."
    - `abstained` — raw cosine below the fitted no-match threshold, fitted for the *null class*.
      "This string probably has no USDA entity at all."

    They are not interchangeable. Raw cosine separates the null class better than the confidence
    curve does (AUC 0.771 vs 0.764), which is why abstention does not reuse the curve. `fdc_id`
    is still reported when abstaining, so a caller can see what would have been picked; treating
    an abstention as "no entity" is the caller's decision.
    """
    confidence = load_curve(curve_path)
    threshold = load_threshold() if abstain_threshold is None else abstain_threshold
    overrides = load_overrides()
    out = []
    for text, fdc_id, cosine in top_picks(db_path, strings):
        # An overridden string keeps its measured cosine — the override corrects WHICH entity is
        # right, and says nothing about how confidently the embedding found it. Overwriting the
        # cosine would launder a hand-assertion into a similarity score and quietly change both
        # the flag and the abstention.
        fdc_id = overrides.get(text, fdc_id)
        score = confidence(cosine)
        out.append((text, fdc_id, cosine, score, score < LOW_CONFIDENCE, cosine < threshold))
    return out


def main() -> None:
    curve, cosines, outcomes = fit_curve()
    path = save_curve(curve)
    print(f"confidence curve fitted on {len(cosines)} tune strings -> {path}")
    print(f"  tune nutritional equivalence: {np.mean(outcomes):.1%}")

    labels = load_labels(Path(DEFAULT_OUT) / "labels.jsonl")
    holdout = {row["normalized_text"] for row in load_sample() if row["split"] == "holdout"}
    con = duckdb.connect(str(DEFAULT_DB), read_only=True)
    try:
        kcal = dict(con.execute(
            "SELECT fdc_id, kcal_per_100g FROM silver.usda_foods").fetchall())
    finally:
        con.close()

    rows = resolve(strings=sorted(holdout))
    gaps, flagged_gaps, kept_gaps = [], [], []
    for text, fdc_id, _, _, flagged, _ in rows:
        truth = labels[text]["fdc_id"]
        if truth == NO_MATCH:
            continue
        gap = error(kcal.get(truth), kcal.get(fdc_id))
        if gap is None:
            continue
        gaps.append(gap)
        (flagged_gaps if flagged else kept_gaps).append(gap)

    def rate(values):
        return np.mean([value < EQUIVALENT for value in values]) if values else 0.0

    print(f"\nholdout, shipped resolver (n={len(gaps)})")
    print(f"  nutritionally equivalent: {rate(gaps):.1%}   median error "
          f"{np.median(gaps):.1%}")
    print(f"  not flagged  n={len(kept_gaps):3}  {rate(kept_gaps):.1%} equivalent")
    print(f"  flagged      n={len(flagged_gaps):3}  {rate(flagged_gaps):.1%} equivalent")
    print("\nThese come from the same single holdout read as docs/er_metrics.md §7 — the")
    print("baseline was already measured there as the comparison arm. No new selection.")


if __name__ == "__main__":
    main()
