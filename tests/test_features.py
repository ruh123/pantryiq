"""Features: each one has to separate the cases that motivated it."""
import pytest

from pantryiq.er.features import (
    FEATURES,
    head_facet,
    plain_facet_share,
    row_features,
    token_jaccard,
)


def test_head_facet_prefers_leading_with_the_food():
    """The 2.3b finding: USDA leads with the food, so `Egg, ...` must beat `Eggnog`, which raw
    cosine gets backwards."""
    assert head_facet("egg", "Egg, whole, raw, fresh") == 1.0
    assert head_facet("egg", "Eggnog") == 0.0


def test_head_facet_matches_across_pluralization():
    assert head_facet("onion", "Onions, raw") == 1.0
    assert head_facet("carrots", "Carrot, baby, raw") == 1.0


def test_head_facet_gives_partial_credit_inside_a_compound_leading_facet():
    assert head_facet("flour", "Wheat flour, white, all-purpose") == 0.5


def test_head_facet_handles_an_empty_string():
    assert head_facet("", "Egg, whole") == 0.0


def test_plain_facet_share_separates_the_ordinary_form_from_a_variant():
    """Facet COUNT is misleading — the ordinary entry often carries more facets. What separates
    them is whether those facets are unremarkable."""
    assert plain_facet_share("Egg, whole, raw, fresh") == 1.0
    assert plain_facet_share("Egg, white, dried") == 0.0
    assert plain_facet_share("Egg, whole, raw, fresh") > plain_facet_share("Egg, white, dried")


def test_a_bare_description_is_maximally_plain():
    assert plain_facet_share("Eggnog") == 1.0


def test_plain_facet_share_is_graded():
    value = plain_facet_share("Beans, snap, green, raw")  # snap+green not plain, raw is
    assert 0.0 < value < 1.0


def test_token_jaccard_is_order_insensitive():
    """A comma-inverted USDA string and a recipe phrase differ mostly by word order."""
    assert token_jaccard("whole raw egg", "Egg, raw, whole") == 1.0
    assert token_jaccard("egg", "Cheese, cheddar") == 0.0


def test_token_jaccard_handles_empty_input():
    assert token_jaccard("", "") == 0.0


def test_row_features_emits_exactly_the_declared_features():
    """The model reads FEATURES in order; a silent mismatch would train on shifted columns."""
    values = row_features("egg", "Egg, whole, raw", 0.9, 0.8, False)

    assert tuple(values) == FEATURES
    assert all(isinstance(value, float) for value in values.values())


def test_deprioritized_is_carried_as_a_number():
    assert row_features("egg", "Babyfood, egg yolk", 0.5, 0.5, True)["is_deprioritized"] == 1.0


def test_cosine_margin_is_zero_for_the_leader_and_negative_otherwise():
    """The decision is relative within a string; the other features are all absolute."""
    assert row_features("egg", "Egg, whole", 0.9, 0.8, False, 0.0)["cosine_margin"] == 0.0
    assert row_features("egg", "Eggnog", 0.7, 0.6, False, -0.2)["cosine_margin"] == -0.2


@pytest.mark.parametrize("feature", [f for f in FEATURES if f != "cosine_margin"])
def test_every_absolute_feature_is_bounded(feature):
    """Unbounded features make a logistic model's coefficients uninterpretable and let one
    candidate dominate; the absolute ones are similarities, shares, or flags. cosine_margin is
    exempt by construction — it is a signed difference."""
    low = row_features("egg", "Cheese, cheddar, low fat", 0.0, 0.0, False)[feature]
    high = row_features("egg", "Egg, whole, raw, fresh", 1.0, 1.0, True)[feature]

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
