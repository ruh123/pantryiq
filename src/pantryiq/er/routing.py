"""Calibration and routing — and the measured finding that no confident subset exists.

The scorer's raw `p(best)` is badly calibrated by construction: `class_weight="balanced"` trains
on a reweighted distribution, so mean p(best) is 0.91 against 0.58 observed nutritional
equivalence. Harmless for *ranking* (a monotone transform cannot reorder candidates within a
string), fatal for *routing*, which compares that number to a threshold. Hence isotonic
calibration on out-of-fold predictions.

Two design choices:

- **The calibration target is `P(nutritionally equivalent)`, not `P(entity correct)`.** The plan
  set thresholds at a ≥95% *entity* precision target; that is not measurable against these labels
  (median 31.6% irreducible kcal ambiguity *within* the correct food). kcal is what downstream
  consumes, so kcal is what routing should protect.
- **Fitted on resolvable strings only.** Scoring a `no-match` gold string as "not equivalent"
  conflates two different failures — picking the wrong entity, and failing to declare the null
  class. The null class is routed by its own floor and reported separately.

## The finding

**There is no threshold that buys a usefully large high-accuracy auto-resolve band.** Raw
p(best) correlates with nutritional equivalence at Spearman +0.13, and the outcome rate by score
quintile is not even monotonic (50%, 57%, 49%, 74%, 61%). At a 90% accuracy target the auto band
holds 2 of 201 strings.

The cause is structural rather than a tuning failure: p(best) is driven mostly by embedding
cosine, and a high cosine says the *food* is right. Nutritional equivalence hinges on the
*facet* — canned or raw, whole or nonfat — which cosine barely distinguishes, since those
variants have nearly identical descriptions.

So this module publishes a **coverage/accuracy trade-off curve** instead of a single threshold,
and the practical consequence is recorded: 2.6's Claude adjudicator is **not an optimization,
it is required**. There is no confident subset to skip it on.

⚠️ The isotonic curve is fit on the same out-of-fold scores the report is computed over, so these
tune numbers are mildly optimistic. The frozen holdout (step 5) tests the whole chain once.

Run:  uv run python -m pantryiq.er.routing
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from pantryiq.er.features import build
from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels
from pantryiq.er.nutrition import EQUIVALENT, error
from pantryiq.er.scoring import matrix, out_of_fold

AUTO_TARGET = 0.90  # what the plan asked for; measured to be unreachable at useful coverage
COVERAGE_POINTS = (0.10, 0.25, 0.50, 0.75, 1.00)


def fit_calibration(scores, outcomes):
    """Isotonic map from raw p(best) to P(outcome). Monotone, so ranking is untouched."""
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, outcomes)


def coverage_curve(scores, outcomes, points=COVERAGE_POINTS):
    """[(coverage, threshold, accuracy above it)] — the trade-off, not a single cut.

    Reported at fixed *coverage* rather than fixed accuracy because the accuracy target the plan
    asked for is unreachable: sweeping accuracy would return an empty band and say nothing about
    what the score can actually do.
    """
    order = np.argsort(-scores)
    ranked = outcomes[order]
    rows = []
    for share in points:
        take = max(1, int(round(share * len(scores))))
        rows.append((take / len(scores), float(scores[order][take - 1]),
                     float(ranked[:take].mean()), take))
    return rows


def best_accuracy_at_target(scores, outcomes, target: float = AUTO_TARGET):
    """(coverage, threshold) of the largest band meeting `target`, or (0.0, None)."""
    order = np.argsort(-scores)
    ranked = outcomes[order]
    best = (0.0, None)
    for take in range(1, len(scores) + 1):
        if ranked[:take].mean() >= target:
            best = (take / len(scores), float(scores[order][take - 1]))
    return best


def nomatch_floor(scores, is_null) -> tuple[float, float, float]:
    """(cut, precision, recall) maximizing null-class F1 — fitted, not guessed."""
    best = (0.0, 0.0, 0.0)
    best_f1 = -1.0
    for cut in np.unique(scores):
        predicted = scores < cut
        if not predicted.sum() or not is_null.sum():
            continue
        precision = float(is_null[predicted].mean())
        recall = float(predicted[is_null].mean())
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best, best_f1 = (float(cut), precision, recall), f1
    return best


def tune_outcomes(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT,
                  split: str = "tune"):
    """Per string in `split`: (text, raw p(best), pick, gold, kcal-equivalent, is_null)."""
    table = build(db_path, gold_dir)
    features, target, groups, fdc_ids = matrix(table, split)
    scores = out_of_fold(features, target, groups)

    gold = {text: record["fdc_id"]
            for text, record in load_labels(Path(gold_dir) / "labels.jsonl").items()}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        kcal = dict(con.execute(
            "SELECT fdc_id, kcal_per_100g FROM silver.usda_foods").fetchall())
    finally:
        con.close()

    rows = []
    for text in dict.fromkeys(groups):
        indices = np.flatnonzero(groups == text)
        best = indices[int(np.argmax(scores[indices]))]
        pick, truth = str(fdc_ids[best]), gold[text]
        gap = None if truth == NO_MATCH else error(kcal.get(truth), kcal.get(pick))
        rows.append((text, float(scores[best]), pick, truth,
                     bool(gap is not None and gap < EQUIVALENT), truth == NO_MATCH))
    return rows


def main() -> None:
    rows = tune_outcomes()
    is_null = np.array([row[5] for row in rows], dtype=bool)
    all_scores = np.array([row[1] for row in rows])

    resolvable = [row for row in rows if not row[5]]
    scores = np.array([row[1] for row in resolvable])
    outcomes = np.array([row[4] for row in resolvable], dtype=float)

    calibrator = fit_calibration(scores, outcomes)
    calibrated = calibrator.predict(scores)
    print(f"tune: {len(rows)} strings ({int(is_null.sum())} no-match, "
          f"{len(resolvable)} resolvable)")
    print(f"calibration: raw p(best) mean {scores.mean():.3f} -> calibrated "
          f"{calibrated.mean():.3f}; observed nutritional equivalence {outcomes.mean():.3f}")

    print("\ncoverage / accuracy trade-off (resolvable strings, ranked by calibrated confidence)")
    print("  coverage      n   threshold   kcal-equivalent")
    for share, threshold, accuracy, take in coverage_curve(calibrated, outcomes):
        print(f"  {share:6.0%}    {take:5}     {threshold:.3f}       {accuracy:6.1%}")

    coverage, threshold = best_accuracy_at_target(calibrated, outcomes)
    print(f"\nlargest band reaching the plan's {AUTO_TARGET:.0%} target: "
          f"{coverage:.1%} of strings" + (f" (threshold {threshold:.3f})" if threshold else ""))
    print("  -> no usefully large confident subset exists. 2.6's adjudicator is required,")
    print("     not an optimization: there is nothing to safely skip it on.")

    cut, precision, recall = nomatch_floor(all_scores, is_null)
    above = np.mean([a > b for a in all_scores[~is_null] for b in all_scores[is_null]])
    print(f"\nnull class: floor < {cut:.4f}  precision {precision:.1%}  recall {recall:.1%}")
    print(f"  P(resolvable scores above a no-match string) = {above:.3f}  "
          "(weak but not useless)")


if __name__ == "__main__":
    main()
