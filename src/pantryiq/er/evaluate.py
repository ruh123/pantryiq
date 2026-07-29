"""Paired comparison statistics — for deciding between two rankers, and for step 5's CIs.

Comparing two rankers by their independent confidence intervals is the wrong test and it is
what produced a false "no difference" here. Both rankers score the *same* strings and agree on
most of them, so the sampling variance of the **difference** is much smaller than the variance
of either rate. Two intervals can overlap across most of their range while the paired difference
is unambiguous.

Two statistics:

- `mcnemar` — the exact test for paired binary outcomes. Only the strings where the two rankers
  *disagree* carry information; under the null each discordant string is a coin flip. This is
  the right test for "did ranker B fix more strings than it broke".
- `paired_bootstrap` — resamples *strings* (not rows) with replacement, recomputing the
  difference each time. Works for continuous outcomes like kcal error where McNemar does not
  apply, and gives the CI on the difference directly.

Resampling by string is deliberate: rows within a string are not independent, so bootstrapping
rows would understate every interval.
"""
from __future__ import annotations

import math

import numpy as np

SEED = 20260729
ITERATIONS = 10_000


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """(a-only correct, b-only correct, exact two-sided p) over paired binary outcomes.

    Concordant strings — both right or both wrong — carry no information about which ranker is
    better and are excluded, which is precisely why this is more sensitive than comparing two
    independent rates over the whole sample.
    """
    a_only = sum(1 for first, second in pairs if first and not second)
    b_only = sum(1 for first, second in pairs if second and not first)
    discordant = a_only + b_only
    if discordant == 0:
        return (0, 0, 1.0)
    smaller = min(a_only, b_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / 2 ** discordant
    return (a_only, b_only, min(1.0, 2 * tail))


def paired_bootstrap(pairs: list[tuple[float, float]], statistic=np.mean,
                     iterations: int = ITERATIONS, seed: int = SEED,
                     ) -> tuple[float, float, float, float]:
    """(observed difference b-a, CI low, CI high, share of resamples where b > a).

    `statistic` is applied to each ranker's values per resample — `np.mean` for a rate,
    `np.median` for a typical error.
    """
    if not pairs:
        return (0.0, 0.0, 0.0, 0.5)
    first = np.array([pair[0] for pair in pairs], dtype=np.float64)
    second = np.array([pair[1] for pair in pairs], dtype=np.float64)
    observed = float(statistic(second) - statistic(first))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(iterations, len(pairs)))
    differences = np.array([statistic(second[row]) - statistic(first[row]) for row in indices])
    low, high = np.percentile(differences, [2.5, 97.5])
    return (observed, float(low), float(high), float(np.mean(differences > 0)))
