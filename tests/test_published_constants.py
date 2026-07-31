"""Pins the constants that define published numbers.

A 50-mutation sweep (er_metrics.md §13) found that changing any of these silently altered a
figure in the README or er_metrics.md while all tests stayed green: the equivalence band, the
kcal floor, the abstention asymmetry, the retrieval-k that sets the recall ceiling, the bootstrap
resolution, the sampling seed that freezes the gold set, and Wilson's z — which turns every
"95% CI" in the docs into an 80% one if it drifts to 1.645.

These are not style assertions. Each value is quoted in a document, so changing one *should*
require editing this file — that edit is the point, and it is what makes a headline change
deliberate rather than accidental.
"""
import inspect

from pantryiq.er import abstain, candidates, evaluate, gold, nutrition, relabel, resolve


def test_the_equivalence_band_is_ten_percent_and_the_bands_agree():
    """`EQUIVALENT` is what resolve, report, entity_map and ensemble all score against, but the
    reported histogram reads its own threshold out of `BANDS` — the two can drift apart and the
    docs would describe one while the headline used the other."""
    assert nutrition.EQUIVALENT == 0.10
    assert nutrition.BANDS[0][0] == nutrition.EQUIVALENT
    assert "within 10%" in nutrition.BANDS[0][1]


def test_the_kcal_floor_is_five_and_is_worth_reporting():
    """The floor credits a pick within 5 kcal/100g regardless of relative error. It is worth
    +2.7pp per unique string, and it is why the metric is labelled "within 10% OR 5 kcal/100g"
    rather than "within 10%" — §11 publishes the sensitivity."""
    assert nutrition.MATERIAL_KCAL == 5.0
    assert nutrition.error(10.0, 14.0) == 0.0      # 40% relative, but 4 kcal apart
    assert nutrition.error(10.0, 16.0) > 0.0       # 6 kcal apart — the floor stops here


def test_abstention_prefers_catching_nulls_over_avoiding_false_abstentions():
    """beta > 1 IS guide rule 4 in code: "a wrong match is worse than an honest gap". beta = 0.5
    reverses the project's stated safety posture while every threshold still fits cleanly."""
    assert abstain.BETA == 2.0


def test_the_retrieval_k_that_sets_the_recall_ceiling():
    """recall@50 = 90.5% is the ceiling on everything downstream — a scorer cannot pick what was
    never retrieved. Every test passes `k` explicitly, so the shipped default was unpinned."""
    assert candidates.TOP_K == 50


def test_the_bootstrap_has_enough_iterations_to_be_stable():
    """Every CI in the docs comes through here. At 5 iterations the intervals are noise and
    nothing else in the suite notices."""
    assert evaluate.ITERATIONS == 10_000


def test_the_gold_sample_seed_is_frozen():
    """The 300-row sample is drawn with this seed. Change it and the frozen tune/holdout split
    silently becomes a different set of strings, orphaning every label and every published
    number — while `verify_sample` is the only thing that would complain."""
    assert gold.SEED == 20260724


def test_wilson_defaults_to_a_genuine_ninety_five_percent_interval():
    """z = 1.645 is the 90% two-sided value; substituting it would relabel every "95% CI" in the
    docs without changing a single caller."""
    assert inspect.signature(relabel.wilson).parameters["z"].default == 1.96


def test_wilson_matches_the_hand_computed_interval_at_n_thirty():
    """Pins the arithmetic, not just the constant. Computed independently from the Wilson score
    formula for p = 27/30, n = 30, z = 1.96:
        centre = (0.9 + 1.96^2/60) / (1 + 1.96^2/30) = 0.854594
        half   = 1.96 * sqrt(0.9*0.1/30 + 1.96^2/3600) / (1 + 1.96^2/30) = 0.110807
    A naive normal approximation gives (0.7926, 1.0074) — outside [0, 1], which is the whole
    reason Wilson is used here."""
    low, high = relabel.wilson(27, 30)

    assert (round(low, 4), round(high, 4)) == (0.7438, 0.9654)


def test_the_low_confidence_flag_threshold():
    """The cut that decides `flagged` in the entity map, quoted in §8 and §11."""
    assert resolve.LOW_CONFIDENCE == 0.40
