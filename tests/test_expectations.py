"""Distribution expectations: advisory checks the dbt gate cannot express."""
from pathlib import Path

import pytest

from pantryiq.gold.expectations import EXPECTATIONS, evaluate


def test_every_expectation_declares_a_reason():
    """A range with no rationale gets widened the first time it fires, which is how a check
    becomes decoration."""
    for label, query, low, high, why in EXPECTATIONS:
        assert low < high, label
        assert len(why) > 30, label
        assert "SELECT" in query.upper(), label


def test_a_tail_statistic_is_present_not_only_central_ones():
    """Measured, not assumed: the pack-size bug (3x mass on 218 of 15,000 recipes) moved p99
    from 4,172 to 12,517 g and left the median at 652.0 g EXACTLY. A suite of means and medians
    would have passed straight through it, so a tail statistic has to be in the list."""
    labels = [label for label, *_ in EXPECTATIONS]

    assert any("p99" in label for label in labels)


@pytest.mark.skipif(not Path("data/pantryiq.duckdb").exists(), reason="corpus not present")
def test_the_expectations_hold_on_the_published_gold():
    """A check that fires on real data gets muted, and a muted check protects nothing."""
    drifted = [row["label"] for row in evaluate() if not row["ok"]]

    assert not drifted, f"drifted: {drifted}"
