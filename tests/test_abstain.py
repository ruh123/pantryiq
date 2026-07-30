"""The fitted no-match threshold. Guide rule 4 makes the errors asymmetric — a wrong match
silently produces wrong nutrition — so the objective is deliberately not F1."""
import numpy as np
import pytest

from pantryiq.er.abstain import (
    cross_validated,
    f_beta,
    fit_threshold,
    load_threshold,
    save,
    sweep,
)


def test_beta_above_one_favours_catching_nulls_over_avoiding_false_abstentions():
    """Rule 4's asymmetry, expressed in the objective rather than in prose."""
    high_recall = f_beta(precision=0.3, recall=0.9, beta=2.0)
    high_precision = f_beta(precision=0.9, recall=0.3, beta=2.0)

    assert high_recall > high_precision
    # F1 is symmetric, which is exactly why it is not the default here.
    assert f_beta(0.3, 0.9, beta=1.0) == pytest.approx(f_beta(0.9, 0.3, beta=1.0))


def test_f_beta_is_defined_when_nothing_is_caught():
    assert f_beta(0.0, 0.0) == 0.0


def test_a_perfectly_separating_signal_is_found():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    is_null = np.array([True, True, False, False])

    threshold = fit_threshold(scores, is_null)

    assert 0.2 < threshold <= 0.8
    assert set(scores[scores < threshold]) == {0.1, 0.2}


def test_a_useless_signal_yields_no_abstention():
    """If the score carries no information, declining on anything only costs coverage."""
    scores = np.array([0.5] * 10)
    is_null = np.array([True, False] * 5)

    assert fit_threshold(scores, is_null) == 0.0


def test_sweep_covers_every_achievable_operating_point():
    scores = np.array([0.1, 0.4, 0.7])
    is_null = np.array([True, False, False])

    rows = sweep(scores, is_null)

    # One row per cut that abstains on something: 0.4 and 0.7 (0.1 abstains on nothing).
    assert [row[0] for row in rows] == [0.4, 0.7]
    assert rows[0][2] == pytest.approx(1.0)   # cut at 0.4 catches the single null
    assert rows[0][3] == pytest.approx(1.0)   # and nothing else


def test_abstaining_on_nothing_is_not_reported_as_an_option():
    """A cut below the minimum score declines on zero strings; scoring it would divide by zero
    and would in any case not be an operating point."""
    scores = np.array([0.5, 0.6])
    is_null = np.array([True, False])

    assert all(row[1] > 0 for row in sweep(scores, is_null))


def test_cross_validation_refits_inside_each_fold():
    """A threshold picked on all 201 strings and scored on those same strings reports its own
    best case. These numbers must be achievable on unseen strings."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0.0, 0.5, 40), rng.uniform(0.5, 1.0, 160)])
    is_null = np.array([True] * 40 + [False] * 160)

    recall, precision, abstain_rate = cross_validated(scores, is_null, folds=4)

    assert recall > 0.8 and precision > 0.8       # a clean signal survives cross-validation
    assert 0.0 < abstain_rate < 1.0


def test_cross_validation_does_not_flatter_a_noise_signal():
    """The check that matters: pure noise must not produce a good out-of-fold score."""
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 1.0, 300)
    is_null = rng.random(300) < 0.12

    recall, precision, _ = cross_validated(scores, is_null, folds=5)

    assert precision < 0.3   # near the 12% base rate, not the in-sample optimum


def test_the_threshold_round_trips_and_defaults_to_never_abstaining(tmp_path):
    """An unfitted threshold must leave the resolver behaving exactly as it did before."""
    assert load_threshold(tmp_path / "absent.json") == 0.0

    path = save(0.644, tmp_path / "t.json", beta=2.0)
    assert load_threshold(path) == pytest.approx(0.644)
