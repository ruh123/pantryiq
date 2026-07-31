"""Servings extraction: only what the recipe actually states, never an estimate."""
import pytest

from pantryiq.silver.servings import parse_servings

# (text, servings) — real RecipeNLG yield phrasings.
CASES = [
    ("Bake 40 minutes. Serves 6.", 6),
    ("Serves 4 to 6.", 4),
    ("Makes 8 servings.", 8),
    ("Makes about 5 servings", 5),
    ("Yields 6", 6),
    ("Yield: 12", 12),
    ("Serves two", 2),
    ("Makes 4 dozen cookies.", 48),          # a dozen-yield is multiplied out
    ("Makes 2 dozen", 24),
    # Yields in units that are not portions — a loaf is not a serving.
    ("Makes 2 loaves.", None),
    ("Makes 1 pie.", None),
    ("Yields 3 quarts.", None),
    ("Makes 2 batches", None),
    # Nothing stated at all.
    ("Bake at 350 for 40 minutes.", None),
    ("Mix well and chill.", None),
    ("", None),
    # Implausible — more likely a misparse than a real yield.
    ("Serves 5000", None),
]


@pytest.mark.parametrize("text,servings", CASES)
def test_parse_servings(text, servings):
    assert parse_servings(text) == servings


def test_a_dozen_yield_is_not_read_as_its_face_value():
    """"Makes 4 dozen cookies" read as 4 servings would overstate per-serving energy twelvefold
    — the single worst way to be wrong about a denominator."""
    assert parse_servings("Makes 4 dozen cookies.") == 48
    assert parse_servings("Makes 4 cookies.") == 4


def test_a_non_portion_yield_is_none_rather_than_a_guess():
    """"Makes 2 loaves" says nothing about how many people it feeds. None is the honest answer;
    2 would be a fabricated denominator that looks like a measurement."""
    assert parse_servings("Makes 2 loaves") is None
    assert parse_servings("Makes 2 servings") == 2
