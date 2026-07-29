"""The 2.5 headline report — the frozen holdout, read once.

Everything is fitted on the 201-string tune split and applied unchanged to the 99-string
holdout: the scorer, the isotonic calibration, and the routing thresholds. Nothing here selects
anything; the ranker choice was made in step 3b on tune evidence and the holdout is the single
clean estimate of what that choice bought.

Reported in the unit the product consumes — relative kcal error — with entity accuracy alongside,
because entity accuracy against these labels is partly a measure of labeling convention (see
`docs/er_metrics.md`). The ambiguity ceiling is reported on the same strings so the numbers can
be read against what is achievable rather than against 100%.

The baseline comparison is legitimate here: both rankers are evaluated on the holdout and
compared *pairwise*, which is a comparison, not a selection.

Run:  uv run python -m pantryiq.er.report
"""
from __future__ import annotations

import statistics
from pathlib import Path

import duckdb
import numpy as np

from pantryiq.er.convention import candidates_by_string
from pantryiq.er.evaluate import bootstrap_ci, mcnemar, paired_bootstrap
from pantryiq.er.features import build
from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels
from pantryiq.er.nutrition import BANDS, EQUIVALENT, ambiguity, distribution, error
from pantryiq.er.routing import fit_calibration
from pantryiq.er.scoring import make_model, matrix, out_of_fold


def _usda(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT fdc_id, description_raw, kcal_per_100g FROM silver.usda_foods").fetchall()
        return {r[0]: r[1] for r in rows}, {r[0]: r[2] for r in rows}
    finally:
        con.close()


def picks(table, split, scores, column="rank"):
    """Per string in `split`: (text, argmax pick by `scores`, baseline pick by lowest rank)."""
    columns = {name: table.column(name).to_pylist()
               for name in ("normalized_text", "fdc_id", "split", column)}
    rows_of: dict[str, list[int]] = {}
    for index, value in enumerate(columns["split"]):
        if value == split:
            rows_of.setdefault(columns["normalized_text"][index], []).append(index)
    offset = {index: position for position, index in
              enumerate(i for i, v in enumerate(columns["split"]) if v == split)}
    out = []
    for text, indices in rows_of.items():
        best = max(indices, key=lambda index: scores[offset[index]])
        baseline = min(indices, key=lambda index: columns[column][index])
        out.append((text, columns["fdc_id"][best], columns["fdc_id"][baseline],
                    float(scores[offset[best]])))
    return out


def main(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT) -> None:
    table = build(db_path, gold_dir)
    description, kcal = _usda(db_path)
    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    gold = {text: record["fdc_id"] for text, record in labels.items()}
    stratum_of = dict(zip(table.column("normalized_text").to_pylist(),
                          table.column("stratum").to_pylist()))
    control_of = dict(zip(table.column("normalized_text").to_pylist(),
                          table.column("control").to_pylist()))

    # --- fit on tune ONLY -------------------------------------------------------------
    tune_x, tune_y, tune_groups, _ = matrix(table, "tune")
    model = make_model().fit(tune_x, tune_y)
    oof = out_of_fold(tune_x, tune_y, tune_groups)
    tune_picks = picks(table, "tune", oof)
    tune_scores, tune_outcomes = [], []
    for text, pick, _, score in tune_picks:
        if gold[text] == NO_MATCH:
            continue
        gap = error(kcal.get(gold[text]), kcal.get(pick))
        if gap is not None:
            tune_scores.append(score)
            tune_outcomes.append(float(gap < EQUIVALENT))
    calibrator = fit_calibration(np.array(tune_scores), np.array(tune_outcomes))

    # --- apply to the holdout, unchanged ---------------------------------------------
    hold_x, hold_y, hold_groups, _ = matrix(table, "holdout")
    hold_scores = model.predict_proba(hold_x)[:, 1]
    holdout = picks(table, "holdout", hold_scores)

    resolvable = [row for row in holdout if gold[row[0]] != NO_MATCH]
    has_gold = {text for text, is_match, split in zip(
        table.column("normalized_text").to_pylist(),
        table.column("is_match").to_pylist(),
        table.column("split").to_pylist()) if split == "holdout" and is_match}

    print("=" * 78)
    print("2.5 HOLDOUT REPORT — frozen 99-string split, read once")
    print("=" * 78)
    print(f"strings: {len(holdout)}   resolvable: {len(resolvable)}   "
          f"no-match: {len(holdout) - len(resolvable)}")
    print(f"retrieval ceiling on this split: {len(has_gold)}/{len(resolvable)} = "
          f"{100 * len(has_gold) / len(resolvable):.1f}%  (gold present in the candidate set)")

    # --- entity accuracy --------------------------------------------------------------
    entity_pairs, kcal_pairs, equiv_pairs, model_errors, strata_rows = [], [], [], [], []
    for text, pick, baseline, _ in resolvable:
        if text not in has_gold:
            continue
        truth = gold[text]
        same = description.get(truth)
        entity_pairs.append((description.get(baseline) == same, description.get(pick) == same))
        gap_model = error(kcal.get(truth), kcal.get(pick))
        gap_base = error(kcal.get(truth), kcal.get(baseline))
        if gap_model is not None and gap_base is not None:
            kcal_pairs.append((gap_base, gap_model))
            equiv_pairs.append((gap_base < EQUIVALENT, gap_model < EQUIVALENT))
            model_errors.append(gap_model)
            strata_rows.append((stratum_of[text], gap_model < EQUIVALENT))

    correct = sum(1 for _, value in entity_pairs if value)
    print("\nENTITY top-1 (rule-6 duplicate credit applied)")
    print(f"  {correct}/{len(entity_pairs)} = {100 * correct / len(entity_pairs):.1f}%  "
          "of strings whose gold was retrieved")
    print(f"  {correct}/{len(holdout)} = {100 * correct / len(holdout):.1f}%  of all holdout "
          "strings (incl. no-match + never-retrieved)")

    # --- the headline: nutrition ------------------------------------------------------
    equivalent = [float(value) for _, value in equiv_pairs]
    rate, low, high = bootstrap_ci(equivalent)
    print(f"\nNUTRITION — relative kcal error of the chosen entity  (n={len(model_errors)})")
    print(f"  nutritionally equivalent (<{EQUIVALENT:.0%}): {int(sum(equivalent))}/"
          f"{len(equivalent)} = {rate:.1%}  95% CI [{low:.1%}, {high:.1%}]")
    counts = distribution(model_errors)
    for _, label in BANDS:
        print(f"    {label:44} {counts[label]:3}")
    median, median_low, median_high = bootstrap_ci(model_errors, statistic=np.median)
    mean, mean_low, mean_high = bootstrap_ci(model_errors)
    print(f"  median {median:.1%} [{median_low:.1%}, {median_high:.1%}]   "
          f"mean {mean:.1%} [{mean_low:.1%}, {mean_high:.1%}]")

    print("\n  by stratum")
    for stratum in ("head", "mid", "tail"):
        subset = [float(value) for name, value in strata_rows if name == stratum]
        if subset:
            share, sub_low, sub_high = bootstrap_ci(subset)
            print(f"    {stratum:5} n={len(subset):3}  {share:5.1%}  "
                  f"[{sub_low:.1%}, {sub_high:.1%}]")

    # --- what was achievable ----------------------------------------------------------
    candidates = candidates_by_string(duckdb.connect(str(db_path), read_only=True),
                                      [row[0] for row in resolvable])
    bounds = [value for text, *_ in resolvable
              if (value := ambiguity(gold[text], candidates.get(text, []), description, kcal))
              is not None]
    if bounds:
        print(f"\n  ambiguity ceiling on these strings: median "
              f"{statistics.median(bounds):.1%} kcal spread WITHIN the correct food "
              f"(n={len(bounds)})")
        print("    — the error a perfect resolver still cannot avoid.")

    # --- baseline comparison, paired --------------------------------------------------
    print("\nVERSUS THE TRIVIAL BASELINE (top embedding cosine), paired on the same strings")
    for name, pairs in (("entity top-1", entity_pairs), ("kcal-equivalent", equiv_pairs)):
        base_only, model_only, p_value = mcnemar(pairs)
        observed, ci_low, ci_high, share = paired_bootstrap(
            [(float(a), float(b)) for a, b in pairs])
        print(f"  {name:16} baseline-only {base_only:2}  model-only {model_only:2}  "
              f"exact p={p_value:.3f}   {100 * observed:+.1f}pp "
              f"[{100 * ci_low:+.1f}, {100 * ci_high:+.1f}]  P(model better)={share:.2f}")
    observed, ci_low, ci_high, share = paired_bootstrap(kcal_pairs, statistic=np.median)
    print(f"  {'median kcal err':16} {100 * observed:+.1f}pp "
          f"[{100 * ci_low:+.1f}, {100 * ci_high:+.1f}]  P(model worse)={share:.2f}")

    # --- anchoring diagnostic ---------------------------------------------------------
    print("\nROUTING BANDS on the holdout, using the tune-fitted calibration")
    calibrated = calibrator.predict(np.array([row[3] for row in resolvable]))
    equivalence = {}
    for row in resolvable:
        gap = error(kcal.get(gold[row[0]]), kcal.get(row[1]))
        equivalence[row[0]] = None if gap is None else gap < EQUIVALENT
    for name, low_edge, high_edge in (("high (>=0.69)", 0.69, 1.01),
                                      ("mid  (0.51-0.69)", 0.506, 0.69),
                                      ("low  (<0.51)", -0.01, 0.506)):
        chosen = [row[0] for row, value in zip(resolvable, calibrated)
                  if low_edge <= value < high_edge]
        outcomes = [equivalence[text] for text in chosen if equivalence[text] is not None]
        if outcomes:
            print(f"  {name:18} n={len(outcomes):3}  kcal-equivalent "
                  f"{sum(outcomes) / len(outcomes):5.1%}")
    print("  Step 4 measured no usefully large high-accuracy band on tune; this is the")
    print("  out-of-sample check of that claim.")

    print("\nANCHORING DIAGNOSTIC — control rows were labeled without a ranked suggestion")
    for flag, name in ((True, "control (unranked)"), (False, "ranked")):
        subset = [equivalence[row[0]] for row in resolvable
                  if control_of.get(row[0], False) == flag and equivalence[row[0]] is not None]
        if subset:
            print(f"  {name:20} n={len(subset):3}  kcal-equivalent "
                  f"{sum(subset) / len(subset):5.1%}")
    print("  A large gap would mean the gold labels partly encode the ranking they were shown.")


if __name__ == "__main__":
    main()
