"""Nutrition error: symmetric, bounded, honest about missing energy values."""
import pytest

from pantryiq.er.nutrition import ambiguity, band_of, distribution, error

DESCRIPTIONS = {
    "1": "Milk, fluid, whole",
    "2": "Milk, fluid, nonfat",
    "3": "Milk, fluid, 2% milkfat",
    "9": "Cream, fluid, heavy whipping",
}
KCAL = {"1": 61.0, "2": 34.0, "3": 50.0, "9": 340.0, "7": None}


def test_error_is_symmetric():
    """Which entity is 'gold' must not change the size of the gap."""
    assert error(134.0, 78.0) == error(78.0, 134.0)


def test_error_is_bounded_by_the_larger_value():
    """Relative-to-gold explodes on near-zero-kcal foods: 32 vs 254 reads as 694%, which says
    nothing useful about a food that is mostly water."""
    assert error(32.0, 254.0) == pytest.approx(0.874, abs=1e-3)
    assert error(0.0, 500.0) == 1.0


def test_a_missing_energy_value_is_not_a_zero_error():
    """73 USDA foods carry no energy at all — that is a real case, not agreement."""
    assert error(None, 100.0) is None
    assert error(100.0, None) is None
    assert error(None, None) is None


def test_two_zero_kcal_foods_agree():
    assert error(0.0, 0.0) == 0.0


def test_a_gap_too_small_to_matter_is_not_an_error():
    """The bug this fixes: `Beverages, tea` spans 0 to 1 kcal/100g, and |0-1|/1 = 100% filed
    brewed tea under "materially wrong". Absolute size has to gate the ratio."""
    assert error(0.0, 1.0) == 0.0
    assert error(16.0, 19.0) == 0.0
    assert error(0.0, 1.0, floor=0.0) == 1.0  # the floor is what suppresses it, not rounding


def test_the_floor_does_not_swallow_a_real_gap():
    assert error(34.0, 61.0) > 0.4


@pytest.mark.parametrize("value,expected", [
    (0.0, "within 10%  (nutritionally equivalent)"),
    (0.099, "within 10%  (nutritionally equivalent)"),
    (0.10, "10-25%"),
    (0.42, "25-50%"),
    (0.87, "over 50%    (materially wrong)"),
    (1.0, "over 50%    (materially wrong)"),
])
def test_bands(value, expected):
    assert band_of(value) == expected


def test_distribution_counts_every_error_once():
    counts = distribution([0.0, 0.05, 0.2, 0.42, 0.9])

    assert sum(counts.values()) == 5
    assert counts["within 10%  (nutritionally equivalent)"] == 2


def test_ambiguity_spans_only_entities_for_the_same_food():
    """Heavy cream shares no food key with milk, so it is not part of milk's ambiguity."""
    candidates = [(fdc_id, DESCRIPTIONS[fdc_id], KCAL[fdc_id], False, 0.9)
                  for fdc_id in ("1", "2", "3", "9")]

    # milk spans 34-61 kcal; including cream at 340 would inflate it fourfold
    assert ambiguity("1", candidates, DESCRIPTIONS, KCAL) == pytest.approx(1 - 34 / 61, abs=1e-6)


def test_ambiguity_is_undefined_with_nothing_to_compare():
    candidates = [("1", DESCRIPTIONS["1"], KCAL["1"], False, 0.9)]

    assert ambiguity("1", candidates, DESCRIPTIONS, KCAL) is None
    assert ambiguity("absent", candidates, DESCRIPTIONS, KCAL) is None


def test_ambiguity_ignores_entities_without_energy():
    candidates = [(fdc_id, DESCRIPTIONS.get(fdc_id, "Milk, fluid, dry"), KCAL[fdc_id], False, 0.9)
                  for fdc_id in ("1", "7")]

    assert ambiguity("1", candidates, DESCRIPTIONS, KCAL) is None
