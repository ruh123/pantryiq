"""A fitted `no-match` threshold — letting the resolver decline instead of guessing.

§11's limitation: the resolver always returns its best candidate, so the ~12% of strings with no
real USDA entity are silently assigned one, producing confident wrong nutrition. The confidence
flag caught 63.9% of them by accident, but that curve was fitted for *nutrition error*, which is
a different target.

Three choices, each measured rather than assumed:

- **The signal is raw cosine, not the calibrated confidence.** On the tune split, raw cosine
  separates the null class at AUC 0.771 against the confidence curve's 0.764 and the top-1/top-2
  margin's 0.598. Margin is the classic abstention signal and it is nearly useless here, for a
  reason specific to this problem: a margin is small when two candidates *compete*, but a
  `no-match` string has no good candidate at all, so its whole candidate set scores low together.
  Absolute similarity is the thing that carries the signal.
- **The objective is F-beta on the null class, with beta > 1 by default.** Guide rule 4 states
  the asymmetry: "a wrong match is worse than an honest gap, because a wrong match silently
  produces wrong nutrition downstream." Beta = 2 weights catching a null twice as heavily as
  avoiding a false abstention. Both the beta = 1 and beta = 2 thresholds are reported so the
  trade-off is visible rather than baked in.
- **The reported performance is cross-validated.** A threshold chosen on 201 strings and scored
  on those same 201 strings reports its own best case. Folds are over strings, and the threshold
  is refitted inside each fold.

Run:  uv run python -m pantryiq.er.abstain
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels, load_sample

THRESHOLD_PATH = Path("data/abstain_threshold.json")
BETA = 2.0  # rule 4: catching a null matters more than avoiding a false abstention
FOLDS = 5
SEED = 20260730


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    if precision + recall == 0:
        return 0.0
    weight = beta * beta
    return (1 + weight) * precision * recall / (weight * precision + recall)


def sweep(scores: np.ndarray, is_null: np.ndarray, beta: float = BETA):
    """[(threshold, abstain_rate, null_recall, null_precision, f_beta)] over observed cuts.

    Abstain when `score < threshold`. Cuts are the observed scores themselves, so every
    achievable operating point is considered exactly once.
    """
    rows = []
    for threshold in np.unique(scores):
        abstained = scores < threshold
        if not abstained.any():
            continue
        precision = float(is_null[abstained].mean())
        recall = float(abstained[is_null].mean()) if is_null.any() else 0.0
        rows.append((float(threshold), float(abstained.mean()), recall, precision,
                     f_beta(precision, recall, beta)))
    return rows


def fit_threshold(scores: np.ndarray, is_null: np.ndarray, beta: float = BETA) -> float:
    """The cut maximizing F-beta on the null class. 0.0 (never abstain) if nothing helps."""
    rows = sweep(scores, is_null, beta)
    if not rows:
        return 0.0
    best = max(rows, key=lambda row: row[4])
    return best[0] if best[4] > 0 else 0.0


def cross_validated(scores: np.ndarray, is_null: np.ndarray, beta: float = BETA,
                    folds: int = FOLDS, seed: int = SEED):
    """(null recall, null precision, abstain rate) from out-of-fold decisions.

    The threshold is refitted inside each fold, so the reported numbers are what a *newly fitted*
    threshold achieves on strings it has not seen — not the in-sample optimum.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scores))
    predicted = np.zeros(len(scores), dtype=bool)
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)
        threshold = fit_threshold(scores[train], is_null[train], beta)
        predicted[test] = scores[test] < threshold
    recall = float(predicted[is_null].mean()) if is_null.any() else 0.0
    precision = float(is_null[predicted].mean()) if predicted.any() else 0.0
    return (recall, precision, float(predicted.mean()))


def tune_signals(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT):
    """(strings, top-1 cosine, is_null) for the tune split."""
    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    tune = {row["normalized_text"] for row in load_sample(gold_dir) if row["split"] == "tune"}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT normalized_text, max(cosine_search) FROM silver.ingredient_candidates
            WHERE normalized_text IN (SELECT UNNEST(?)) GROUP BY 1 ORDER BY 1
            """,
            [sorted(tune)],
        ).fetchall()
    finally:
        con.close()
    texts = [row[0] for row in rows]
    return (texts,
            np.array([row[1] for row in rows], dtype=float),
            np.array([labels[text]["fdc_id"] == NO_MATCH for text in texts], dtype=bool))


def save(threshold: float, path: Path | str = THRESHOLD_PATH, **context) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"threshold": threshold, "signal": "max cosine_search",
                                "fitted_on": "tune split", **context}, indent=2) + "\n",
                    encoding="utf-8")
    return path


def load_threshold(path: Path | str = THRESHOLD_PATH) -> float:
    """The fitted cut, or 0.0 (never abstain) when it has not been fitted yet."""
    path = Path(path)
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


def main() -> None:
    texts, scores, is_null = tune_signals()
    print(f"tune split: {len(texts)} strings, {int(is_null.sum())} no-match "
          f"({100 * is_null.mean():.1f}%)\n")

    print("operating points (abstain when max cosine < threshold)")
    print(f"  {'threshold':>10} {'abstain':>9} {'null recall':>12} {'null prec':>10} "
          f"{'resolved purity':>16}")
    rows = sweep(scores, is_null)
    for target in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        nearest = min(rows, key=lambda row: abs(row[1] - target))
        threshold, abstain_rate, recall, precision, _ = nearest
        kept = scores >= threshold
        purity = 100 * (1 - is_null[kept].mean()) if kept.any() else 100.0
        print(f"  {threshold:10.3f} {100 * abstain_rate:8.1f}% {100 * recall:11.1f}% "
              f"{100 * precision:9.1f}% {purity:15.1f}%")

    baseline = 100 * (1 - is_null.mean())
    print(f"\n  resolved purity with no abstention at all: {baseline:.1f}%")

    for beta, label in ((1.0, "F1  (neutral)"), (BETA, f"F{BETA:.0f}  (rule 4: prefer abstaining)")):
        threshold = fit_threshold(scores, is_null, beta)
        recall, precision, abstain_rate = cross_validated(scores, is_null, beta)
        kept = scores >= threshold
        purity = 100 * (1 - is_null[kept].mean()) if kept.any() else 100.0
        print(f"\n{label}: threshold {threshold:.3f}")
        print(f"  cross-validated: null recall {100 * recall:.1f}%, "
              f"null precision {100 * precision:.1f}%, abstains on {100 * abstain_rate:.1f}%")
        print(f"  in-sample resolved purity {purity:.1f}% "
              f"(from {baseline:.1f}% with no abstention)")

    chosen = fit_threshold(scores, is_null, BETA)
    recall, precision, abstain_rate = cross_validated(scores, is_null, BETA)
    path = save(chosen, beta=BETA,
                cv_null_recall=round(recall, 4), cv_null_precision=round(precision, 4),
                cv_abstain_rate=round(abstain_rate, 4))
    print(f"\nsaved F{BETA:.0f} threshold {chosen:.3f} -> {path}")
    print("The old flag used the nutrition-fitted confidence curve (AUC 0.764); raw cosine "
          "separates\nthe null class better (AUC 0.771) and is what this threshold applies to.")


if __name__ == "__main__":
    main()
