"""Unit tests for the RecipeNLG inspection helpers (no dataset required)."""
from inspect_recipenlg import QTY, parse_list


def test_parse_list_json():
    assert parse_list('["a", "b"]') == ["a", "b"]


def test_parse_list_python_literal():
    assert parse_list("['a', 'b']") == ["a", "b"]


def test_parse_list_non_list_returns_none():
    assert parse_list('"just a string"') is None
    assert parse_list("definitely not a list") is None


def test_qty_matches_quantities():
    for line in [
        "1 c. firmly packed brown sugar",
        "1/2 tsp. vanilla",
        "2 1/2 cups flour",
        "1.5 oz cream cheese",
        "¾ cup sugar",
        "⅙ tsp nutmeg",
        "a pinch of salt",
        "one egg",
        "half a lemon",
    ]:
        assert QTY.match(line), line


def test_qty_rejects_non_quantities():
    for line in [
        "2% milk",
        "100% whole wheat flour",
        "salt to taste",
        "fresh basil",
        "cream of mushroom soup",
    ]:
        assert not QTY.match(line), line
