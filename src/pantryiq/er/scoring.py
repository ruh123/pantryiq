"""Pairwise scorer for entity resolution — fit on the tune split only.

A logistic regression over the eight features in `features.py`, scoring each (string, candidate)
pair independently; the decision per string is then `argmax` over its candidates, and the
winner's calibrated probability is what routing thresholds apply to.

Three choices here are load-bearing:

- **Cross-validation is grouped by ingredient string** (`GroupKFold`). A plain k-fold would put
  candidates of the same string in both train and test, and within a string the features are
  strongly correlated — that leaks and inflates every number. With ~77 candidates per string,
  the leak would be severe rather than marginal.
- **The reported metric is per-string top-1 accuracy, not pair-level AUC.** 98.9% of pairs are
  negative, so a model that says "no" to everything scores superbly on any pair-level metric and
  resolves nothing. The product question is "did the top-ranked candidate turn out to be the gold
  entity", and that is what is measured.
- **Calibration is measured on `p(best)`, not on individual pairs.** Routing compares the
  winner's probability against a threshold, so that is the number whose reliability matters.

The tune split is 201 strings / 15,524 pairs / 176 positives. Every number here is
cross-validated *within* tune; the frozen 99-string holdout is not touched until step 5.

Run:  uv run python -m pantryiq.er.scoring
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from pantryiq.er.features import FEATURES, build

SEED = 20260729
FOLDS = 5


def matrix(table: pa.Table, split: str | None = None):
    """(X, y, groups, fdc_ids) for a split. Rows keep their table order."""
    keep = (range(table.num_rows) if split is None else
            [index for index, value in enumerate(table.column("split").to_pylist())
             if value == split])
    columns = {name: table.column(name).to_pylist() for name in
               (*FEATURES, "is_match", "normalized_text", "fdc_id")}
    features = np.array([[columns[name][index] for name in FEATURES] for index in keep],
                        dtype=np.float64)
    return (features,
            np.array([columns["is_match"][index] for index in keep], dtype=bool),
            np.array([columns["normalized_text"][index] for index in keep]),
            np.array([columns["fdc_id"][index] for index in keep]))


def make_model():
    """Standardize then fit. `balanced` weights matter: positives are 1.14% of rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                     random_state=SEED)),
    ])


def out_of_fold(features, target, groups, folds: int = FOLDS):
    """Out-of-fold probabilities, grouped by string so no string spans train and test."""
    from sklearn.model_selection import GroupKFold

    probabilities = np.zeros(len(target), dtype=np.float64)
    for train, test in GroupKFold(n_splits=folds).split(features, target, groups):
        model = make_model().fit(features[train], target[train])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
    return probabilities


def per_string_winner(probabilities, target, groups, fdc_ids):
    """One row per string: (text, p(best), whether best is the gold entity, gold retrievable)."""
    winners = []
    for text in dict.fromkeys(groups):
        rows = np.flatnonzero(groups == text)
        best = rows[int(np.argmax(probabilities[rows]))]
        winners.append((text, float(probabilities[best]), bool(target[best]),
                        bool(target[rows].any()), str(fdc_ids[best])))
    return winners


def top1_accuracy(winners) -> tuple[int, int, int]:
    """(correct, retrievable, all) — retrievable excludes strings whose gold was never shown.

    Both denominators are reported: against `retrievable` it measures the scorer, against `all`
    it measures the pipeline, and quoting only the first would flatter it.
    """
    retrievable = [row for row in winners if row[3]]
    return (sum(1 for row in retrievable if row[2]), len(retrievable), len(winners))


def reliability(winners, bins: int = 5):
    """Reliability of p(best): predicted confidence vs observed correctness, per bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = [row for row in winners
                  if low <= row[1] < high or (high == 1.0 and row[1] == 1.0)]
        if chosen:
            rows.append((low, high, len(chosen),
                         float(np.mean([row[1] for row in chosen])),
                         float(np.mean([row[2] for row in chosen]))))
    return rows


def coefficients(features, target) -> list[tuple[str, float]]:
    """Fitted weights on standardized features — comparable to each other in magnitude."""
    model = make_model().fit(features, target)
    weights = model.named_steps["model"].coef_[0]
    return sorted(zip(FEATURES, (float(value) for value in weights)),
                  key=lambda pair: -abs(pair[1]))


def main() -> None:
    table = build()
    features, target, groups, fdc_ids = matrix(table, "tune")
    print(f"tune split: {len(target):,} pairs  {len(set(groups))} strings  "
          f"{int(target.sum())} positives ({100 * target.mean():.2f}%)")

    probabilities = out_of_fold(features, target, groups)
    winners = per_string_winner(probabilities, target, groups, fdc_ids)
    correct, retrievable, total = top1_accuracy(winners)

    print(f"\nout-of-fold top-1 accuracy ({FOLDS}-fold, grouped by string)")
    print(f"  {correct}/{retrievable} = {100 * correct / retrievable:.1f}%  "
          "of strings whose gold entity was retrieved  (measures the scorer)")
    print(f"  {correct}/{total} = {100 * correct / total:.1f}%  "
          "of all tune strings  (measures the pipeline, incl. no-match + never-retrieved)")

    print("\nreliability of p(best) — routing thresholds apply to this number")
    print("  bin           n   mean p(best)   observed correct")
    for low, high, count, predicted, observed in reliability(winners):
        print(f"  {low:.1f}-{high:.1f}  {count:5}      {predicted:6.3f}        {observed:6.3f}")

    print("\ncoefficients on standardized features (full tune fit)")
    for name, weight in coefficients(features, target):
        print(f"  {name:20} {weight:+7.3f}")


if __name__ == "__main__":
    main()
