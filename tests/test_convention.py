"""Rule 2b: fires only on attributes the line was silent about, and only where another entry
for the same food stated fewer of them — where the labeler actually faced a choice."""
import pytest

from pantryiq.er.convention import food_key, had_a_fork, mentions, violations


def test_a_qualifier_the_line_never_asked_for_is_a_violation():
    assert violations("milk", "Milk, nonfat, fluid") == ["fat"]
    assert violations("spiral pasta", "Pasta, dry, unenriched") == ["fortification"]
    assert violations("whole tomato", "Tomatoes, red, ripe, canned") == ["form"]


def test_a_qualifier_the_line_does_ask_for_is_not_a_violation():
    """Only silence is governed — `canned` is stated by the line here."""
    assert violations("canned tomato", "Tomatoes, red, ripe, canned") == []


def test_the_line_speaking_at_all_settles_the_attribute():
    """A line saying `skim` has chosen a fat level, so a `nonfat` entry answers it. Likewise
    `dried` and `dehydrated` are the same claim about form, spelled differently."""
    assert violations("skim milk", "Milk, nonfat, fluid") == []
    assert violations("dried onion", "Onions, dehydrated flakes") == []


def test_a_contradicted_qualifier_is_still_caught():
    """Line says whole, entry says nonfat — rule 2 territory, but it must not pass silently."""
    assert violations("whole milk", "Milk, nonfat, fluid") == ["fat"]


def test_terms_match_on_word_boundaries():
    assert mentions("cream, light whipping", "light")
    assert not mentions("turkish delight", "light")
    assert violations("delight bar", "Candies, delight, plain") == []


def test_several_attributes_can_break_at_once():
    assert violations("green bean", "Beans, snap, green, canned, low fat") == ["fat", "form"]


def test_the_line_stating_a_hyphenated_qualifier_is_not_silence():
    """USDA writes `low fat`, recipes write `low-fat`; treating those as different terms made
    a line that states its fat level look silent, and flagged a correct label."""
    assert violations("low-fat cottage cheese", "Cheese, cottage, lowfat, 1% milkfat") == []


def test_the_fat_vocabulary_covers_percentage_forms():
    """`2% fat` is a fat level. Missing it let the audit 'repair' evaporated milk INTO a 2%
    product — the exact downgrade rule 2b exists to prevent."""
    assert violations("evaporated milk", "Milk, evaporated, 2% fat") == ["fat"]


def test_no_fork_means_no_finding():
    """USDA carries no raw chervil, so `dried` was never a choice the labeler made."""
    candidates = [("1", "Spices, chervil, dried", 237.0, False, 0.9)]

    assert violations("chervil", "Spices, chervil, dried") == ["form"]
    assert not had_a_fork("chervil", "Spices, chervil, dried", candidates)


def test_a_fork_is_another_entry_for_the_same_food_stating_less():
    """Every evaporated milk in USDA is `canned`, so `canned` is inherent — but the fat level
    was a real choice, and that is enough to flag."""
    current = "Milk, canned, evaporated, nonfat"
    candidates = [
        ("1", current, 78.0, False, 0.90),
        ("2", "Milk, canned, evaporated, with added vitamin A", 134.0, False, 0.88),
    ]

    assert violations("evaporated milk", current) == ["fat", "form"]
    assert had_a_fork("evaporated milk", current, candidates)


def test_a_different_food_is_not_a_fork():
    """The failure this guards: `Beans, liquid from stewed kidney beans` leads with `Beans`
    exactly like the real entry, so a one-facet key accepts it as the same food."""
    current = "Beans, kidney, all types, mature seeds, cooked, boiled, with salt"
    candidates = [
        ("1", current, 127.0, False, 0.90),
        ("2", "Beans, liquid from stewed kidney beans", 9.0, False, 0.88),
    ]

    assert not had_a_fork("kidney bean", current, candidates)
    assert had_a_fork("kidney bean", current,
                      [*candidates, ("3", "Beans, kidney, all types, mature seeds, raw",
                                     333.0, False, 0.87)])


def test_a_candidate_that_trades_one_violation_for_another_is_not_a_fork():
    """Strict subset, not merely different: a swap is another opinion, not a plainer entry."""
    current = "Beans, snap, green, canned"
    candidates = [
        ("1", current, 21.0, False, 0.90),
        ("2", "Beans, snap, green, low fat, frozen", 30.0, False, 0.88),
    ]

    assert not had_a_fork("green bean", current, candidates)


@pytest.mark.parametrize("description,expected", [
    ("Milk, canned, evaporated", ("milk", "canned")),
    ("Beans, liquid from stewed kidney beans", ("beans", "liquid from stewed kidney beans")),
    ("Eggnog", ("eggnog",)),
])
def test_food_key_uses_two_leading_facets(description, expected):
    assert food_key(description) == expected
