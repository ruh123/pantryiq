"""Paired statistics: sensitive where independent intervals are not, and honest at small n."""
import numpy as np

from pantryiq.er.evaluate import bootstrap_ci, mcnemar, paired_bootstrap


def test_mcnemar_ignores_strings_where_both_rankers_agree():
    """Concordant pairs carry no information about which ranker is better — including them is
    what makes an unpaired comparison insensitive."""
    agree_right = [(True, True)] * 100
    agree_wrong = [(False, False)] * 100

    assert mcnemar(agree_right + agree_wrong) == (0, 0, 1.0)


def test_mcnemar_detects_a_one_sided_improvement():
    """10 fixed, 0 broken is decisive even though the sample is tiny."""
    pairs = [(False, True)] * 10 + [(True, True)] * 50

    a_only, b_only, p = mcnemar(pairs)

    assert (a_only, b_only) == (0, 10)
    assert p < 0.01


def test_mcnemar_is_symmetric_under_swapping_the_rankers():
    pairs = [(False, True)] * 7 + [(True, False)] * 2
    swapped = [(second, first) for first, second in pairs]

    a_only, b_only, p = mcnemar(pairs)
    swapped_a, swapped_b, swapped_p = mcnemar(swapped)

    assert (a_only, b_only) == (swapped_b, swapped_a)
    assert p == swapped_p


def test_mcnemar_finds_no_effect_when_fixes_and_breaks_balance():
    pairs = [(False, True)] * 8 + [(True, False)] * 8

    _, _, p = mcnemar(pairs)

    assert p > 0.9


def test_paired_bootstrap_brackets_a_real_difference():
    pairs = [(0.0, 1.0)] * 40  # b better on every string

    observed, low, high, share = paired_bootstrap(pairs, iterations=500)

    assert observed == 1.0
    assert low > 0.0 and high >= 1.0
    assert share == 1.0


def test_paired_bootstrap_straddles_zero_when_there_is_no_difference():
    rng = np.random.default_rng(0)
    values = rng.normal(size=60)
    pairs = [(float(value), float(value)) for value in values]

    observed, low, high, share = paired_bootstrap(pairs, iterations=500)

    assert observed == 0.0
    assert low == 0.0 == high  # identical inputs: the difference has no spread at all


def test_paired_bootstrap_accepts_a_median_statistic():
    """kcal error is skewed, so the typical case needs a median, not a mean."""
    pairs = [(0.5, 0.0)] * 30 + [(0.5, 0.9)] * 5

    observed, _, _, share = paired_bootstrap(pairs, statistic=np.median, iterations=500)

    assert observed == -0.5  # b's median is 0.0, a's is 0.5
    assert share < 0.5


def test_paired_bootstrap_handles_an_empty_comparison():
    assert paired_bootstrap([]) == (0.0, 0.0, 0.0, 0.5)


def test_bootstrap_ci_brackets_the_statistic_and_is_reproducible():
    """Produces §7's [53.2, 74.0]-style intervals and had no direct test — it was imported by
    name nowhere in this file. A seeded resample must be deterministic, or two runs of the same
    report disagree about their own confidence intervals."""
    values = [0.0] * 30 + [1.0] * 70

    observed, low, high = bootstrap_ci(values, iterations=2000)

    assert observed == 0.70
    assert low < observed < high
    assert bootstrap_ci(values, iterations=2000) == bootstrap_ci(values, iterations=2000)


def test_bootstrap_ci_narrows_as_the_sample_grows():
    """The property that makes an interval mean anything. A mutation dropping ITERATIONS to 5
    leaves this readable but the interval becomes noise — hence the constant is pinned too."""
    small = bootstrap_ci([0.0, 1.0] * 5, iterations=2000)
    large = bootstrap_ci([0.0, 1.0] * 250, iterations=2000)

    assert (large[2] - large[1]) < (small[2] - small[1])
