"""The stratified estimator behind the 2.7 headline. The gold set is 100/100/100 over strata of
545/1,578/7,201 strings, so an unweighted average answers the wrong question entirely."""
import pytest

from pantryiq.er.entity_map import stratified_ci, stratified_rate

# head is small but occurrence-heavy; tail is the opposite — the corpus's actual shape.
TOTALS = {"head": (100, 1000), "tail": (900, 100)}


def test_the_two_denominators_can_disagree_completely():
    """Same judgments, opposite conclusions. This is why the brief asks for both: the head is
    5.8% of the vocabulary and 83.5% of what a user actually hits."""
    by_stratum = {"head": [(10.0, True)], "tail": [(1.0, False)]}

    per_string = stratified_rate(by_stratum, TOTALS, weighted=False)
    per_occurrence = stratified_rate(by_stratum, TOTALS, weighted=True)

    assert per_string == pytest.approx(0.10)        # 900 of 1000 strings are tail, and tail fails
    assert per_occurrence == pytest.approx(1000 / 1100)  # but head carries the occurrences


def test_reweighting_matters_only_when_the_strata_differ():
    """With equal rates everywhere, the weighting scheme cannot change the answer — a sanity
    check that the estimator is not inventing signal."""
    by_stratum = {"head": [(5.0, True), (5.0, False)], "tail": [(1.0, True), (1.0, False)]}

    assert stratified_rate(by_stratum, TOTALS, weighted=False) == pytest.approx(0.5)
    assert stratified_rate(by_stratum, TOTALS, weighted=True) == pytest.approx(0.5)


def test_an_oversampled_stratum_does_not_dominate():
    """The gold set has 100 tail strings out of 7,201, and 100 head out of 545 — a raw average
    over the sample would weight the tail ~13x too heavily."""
    by_stratum = {"head": [(1.0, True)] * 100, "tail": [(1.0, False)] * 100}

    # A naive pooled average would be 50%. Weighted by true stratum size it is 10%.
    assert stratified_rate(by_stratum, TOTALS, weighted=False) == pytest.approx(0.10)


def test_occurrence_weighting_applies_within_a_stratum_too():
    """One very frequent string should outweigh several rare ones in the same stratum."""
    by_stratum = {"head": [(99.0, True), (1.0, False)]}

    assert stratified_rate(by_stratum, {"head": (2, 100)}, weighted=True) == pytest.approx(0.99)
    assert stratified_rate(by_stratum, {"head": (2, 100)}, weighted=False) == pytest.approx(0.50)


def test_empty_and_missing_strata_are_skipped_not_counted_as_failures():
    """A stratum with no sampled strings carries no information; scoring it as 0% would drag the
    estimate toward zero for a reason that has nothing to do with the pipeline."""
    by_stratum = {"head": [(1.0, True)], "tail": []}

    assert stratified_rate(by_stratum, TOTALS, weighted=False) == pytest.approx(1.0)
    assert stratified_rate({}, TOTALS, weighted=False) == 0.0


def test_a_stratum_absent_from_totals_is_ignored():
    by_stratum = {"head": [(1.0, True)], "unknown": [(1.0, False)]}

    assert stratified_rate(by_stratum, TOTALS, weighted=False) == pytest.approx(1.0)


def test_the_interval_brackets_the_estimate_and_widens_with_less_data():
    """Resampling happens WITHIN each stratum, matching how the sample was drawn."""
    many = {"head": [(1.0, True)] * 40 + [(1.0, False)] * 40,
            "tail": [(1.0, True)] * 40 + [(1.0, False)] * 40}
    few = {"head": [(1.0, True), (1.0, False)], "tail": [(1.0, True), (1.0, False)]}

    estimate, low, high = stratified_ci(many, TOTALS, weighted=False, iterations=400)
    _, narrow_low, narrow_high = stratified_ci(few, TOTALS, weighted=False, iterations=400)

    assert low <= estimate <= high
    assert (high - low) < (narrow_high - narrow_low)


def test_the_interval_is_reproducible():
    by_stratum = {"head": [(1.0, True), (1.0, False)], "tail": [(1.0, True)]}

    assert (stratified_ci(by_stratum, TOTALS, weighted=True, iterations=200)
            == stratified_ci(by_stratum, TOTALS, weighted=True, iterations=200))
