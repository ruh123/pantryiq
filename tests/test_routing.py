"""Routing: the trade-off curve is honest, and an unreachable target reports as unreachable."""
import numpy as np

from pantryiq.er.routing import (
    best_accuracy_at_target,
    coverage_curve,
    fit_calibration,
    nomatch_floor,
)


def test_calibration_maps_inflated_scores_onto_observed_rates():
    """Raw p(best) averages 0.92 against 0.58 observed — routing on the raw number is nonsense."""
    scores = np.linspace(0.85, 0.99, 50)
    outcomes = np.array([0.0] * 25 + [1.0] * 25)

    calibrated = fit_calibration(scores, outcomes).predict(scores)

    assert abs(calibrated.mean() - outcomes.mean()) < 0.05


def test_calibration_is_monotone_so_ranking_survives_it():
    """A calibration that reordered candidates would silently change the resolver's picks."""
    scores = np.array([0.1, 0.4, 0.5, 0.7, 0.9])
    calibrated = fit_calibration(scores, np.array([0.0, 0.0, 1.0, 1.0, 1.0])).predict(scores)

    assert list(calibrated) == sorted(calibrated)


def test_coverage_curve_reports_the_requested_coverage_levels():
    scores = np.linspace(0, 1, 100)
    outcomes = (scores > 0.5).astype(float)

    rows = coverage_curve(scores, outcomes, points=(0.10, 0.50, 1.00))

    assert [row[3] for row in rows] == [10, 50, 100]
    assert rows[0][2] == 1.0     # the top 10% are all positive
    assert rows[-1][2] == 0.5    # everything is the base rate


def test_coverage_curve_ranks_by_score_not_by_table_order():
    scores = np.array([0.1, 0.9, 0.2])
    outcomes = np.array([0.0, 1.0, 0.0])

    top = coverage_curve(scores, outcomes, points=(0.34,))[0]

    assert top[2] == 1.0  # the single highest-scoring row is the positive one


def test_an_unreachable_target_reports_zero_coverage():
    """The finding this protects: at a 90% target the band held 2 of 201 strings. A silent
    fallback to 'best available' would have hidden that."""
    scores = np.linspace(0, 1, 50)
    outcomes = np.zeros(50)

    assert best_accuracy_at_target(scores, outcomes, target=0.90) == (0.0, None)


def test_a_reachable_target_returns_the_largest_qualifying_band():
    scores = np.linspace(0, 1, 100)
    outcomes = (scores > 0.5).astype(float)

    coverage, threshold = best_accuracy_at_target(scores, outcomes, target=0.95)

    assert 0.45 <= coverage <= 0.52
    assert threshold is not None


def test_nomatch_floor_finds_a_separating_cut():
    scores = np.concatenate([np.full(20, 0.2), np.full(80, 0.9)])
    is_null = np.array([True] * 20 + [False] * 80)

    cut, precision, recall = nomatch_floor(scores, is_null)

    assert 0.2 < cut <= 0.9
    assert precision == 1.0 and recall == 1.0


def test_nomatch_floor_is_degenerate_when_there_is_nothing_to_separate():
    scores = np.full(30, 0.5)
    is_null = np.array([True] * 15 + [False] * 15)

    cut, precision, recall = nomatch_floor(scores, is_null)

    assert (cut, precision, recall) == (0.0, 0.0, 0.0)
