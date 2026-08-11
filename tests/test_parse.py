"""Query parsing — the boundary where a model is allowed to influence what gets retrieved.

No test here calls the API. The prompt's behaviour is exercised by `uv run python -m
pantryiq.agent.parse`; what is pinned here is everything that decides whether a *wrong* model
answer can still do damage — the vocabulary boundary, the refusal path, and the fact that the
parser cannot emit anything but a filter.
"""
import json
from types import SimpleNamespace

import pytest

from pantryiq.agent.claude import Refused, injection_paragraph, text_of
from pantryiq.agent.parse import (
    DISH_MIN_COVERAGE,
    FILTER_SCHEMA,
    SYSTEM,
    TAGS,
    build_query,
    parse,
)
from pantryiq.agent.retrieval import DEFAULT_MIN_COVERAGE


def fake_client(payload: dict | str, stop_reason: str = "end_turn"):
    """A client whose one call returns `payload` as the model's structured output."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    blocks = [] if stop_reason == "refusal" else [SimpleNamespace(type="text", text=text)]
    response = SimpleNamespace(stop_reason=stop_reason, content=blocks)
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))


def payload(**overrides) -> dict:
    fields = {"pantry": ["chicken"], "exclude": [], "max_kcal": None, "min_kcal": None,
              "kcal_basis": "total", "tags": [], "max_cost_usd": None}
    fields.update(overrides)
    return fields


def test_a_tag_outside_the_warehouse_vocabulary_is_dropped():
    """The model can invent a plausible tag — 'keto', 'dairy-free', 'paleo'. Passed through, it
    filters to zero rows, and the user reads that as 'we have no keto recipes' rather than 'we do
    not track keto'. Dropping it answers the rest of the question honestly instead."""
    query = build_query(payload(tags=["vegan", "keto", "dairy-free"]))

    assert query.tags == ("vegan",)


def test_an_unknown_kcal_basis_falls_back_to_the_total():
    """`kcal_basis` selects a SQL column. A value outside the enum must not reach retrieval, which
    raises on it — the parser is the wrong place to turn a model slip into a crash."""
    assert build_query(payload(kcal_basis="per_ounce")).kcal_basis == "total"
    assert build_query(payload(kcal_basis=None)).kcal_basis == "total"


def test_pantry_terms_are_lowercased_and_stripped():
    """Matching is a lowercase regex over ingredient text, so 'Chicken ' would never match."""
    query = build_query(payload(pantry=["  Chicken ", "RICE", ""]))

    assert query.pantry == ("chicken", "rice")


def test_the_parser_can_only_emit_a_filter_never_recipe_ids_or_sql():
    """The property the whole design rests on: whatever the question says, the model's output is
    a fixed set of filter fields. There is no field it could put a recipe id or a query in."""
    assert FILTER_SCHEMA["additionalProperties"] is False
    assert set(FILTER_SCHEMA["properties"]) == set(FILTER_SCHEMA["required"])
    assert set(FILTER_SCHEMA["required"]) == {
        "dish", "pantry", "exclude", "max_kcal", "min_kcal", "kcal_basis", "tags", "max_cost_usd"}


def test_the_tag_enum_is_closed_in_the_schema_itself():
    """Belt and braces with the drop in build_query: constraining the schema means the model is
    usually not able to produce the invalid value in the first place."""
    assert FILTER_SCHEMA["properties"]["tags"]["items"]["enum"] == list(TAGS)


def test_the_question_is_fenced_as_untrusted_data():
    """The user's text reaches the prompt verbatim, so the system prompt has to name the element
    and say its contents are not addressed to the model."""
    assert injection_paragraph("user_question", "question") in SYSTEM
    assert "<user_question>" in SYSTEM


def test_a_refusal_raises_rather_than_reading_an_empty_content_array():
    """Opus 5's classifiers decline with HTTP 200 and an EMPTY content list, so the natural
    `response.content[0].text` raises IndexError and reports a bug that is not there."""
    with pytest.raises(Refused):
        parse("anything", fake_client(payload(), stop_reason="refusal"))


def test_a_response_with_no_text_block_is_refused_not_indexed():
    """Same failure by a different route: a stop_reason we do not recognise, with no text."""
    empty = SimpleNamespace(stop_reason="max_tokens", content=[])

    with pytest.raises(Refused):
        text_of(empty)


def test_parse_returns_a_query_the_retriever_can_run():
    query = parse("chicken please", fake_client(
        payload(pantry=["chicken"], max_kcal=500, kcal_basis="serving", tags=["vegetarian"])))

    assert query.pantry == ("chicken",)
    assert (query.max_kcal, query.kcal_basis) == (500, "serving")
    assert query.tags == ("vegetarian",)


def test_the_coverage_bar_is_not_something_the_model_can_move():
    """`min_coverage` is the quality floor for every answer. It is not in the schema, so no
    question — however phrased — can lower it."""
    assert "min_coverage" not in FILTER_SCHEMA["properties"]

    assert parse("x", fake_client(payload())).min_coverage == 0.8


def test_the_tag_vocabulary_still_matches_what_gold_publishes(fixture_gold):
    """Pinned constants drift. If `models/gold/recipe_tags.sql` gains or retires a tag, the
    parser would go on offering a filter that matches nothing, or stop offering a real one."""
    published = {tag for (tag,) in fixture_gold.execute(
        "SELECT DISTINCT tag FROM gold.recipe_tags").fetchall()}

    assert published == set(TAGS)


# --- dish lookup ----------------------------------------------------------------------------------


def test_a_dish_is_its_own_field_and_never_a_pantry_term():
    """`pantry` matches ingredient TEXT, so "lasagna" there searches for an ingredient called
    lasagna and finds nothing. Before `dish` existed the system prompt told the model to drop the
    name entirely, so "chicken pot pie" retrieved on an empty filter and returned taco sauce."""
    query = build_query(payload(dish="Chicken Pot Pie", pantry=[]))

    assert query.dish == "Chicken Pot Pie"
    assert query.pantry == ()


def test_naming_a_dish_drops_the_coverage_bar():
    """Measured, not assumed: of 39 recipes titled pot pie only 2 clear 0.8 coverage, and of 68
    lasagnas only 10. Keeping the bar would answer a request the warehouse can plainly satisfy
    with "nothing matches". Coverage still travels with every figure."""
    assert build_query(payload(dish="lasagna")).min_coverage == DISH_MIN_COVERAGE
    assert DISH_MIN_COVERAGE < DEFAULT_MIN_COVERAGE


def test_a_question_with_no_dish_keeps_the_quality_bar():
    """The bar is the default for nutrition questions and must not be lowered for everyone."""
    assert build_query(payload(pantry=["chicken"])).min_coverage == DEFAULT_MIN_COVERAGE


def test_a_blank_or_missing_dish_is_not_a_dish():
    """A whitespace dish would sanitise to no patterns and, without this, return zero rows —
    reading as "we do not have that" for a question that named nothing."""
    for value in (None, "", "   "):
        query = build_query(payload(dish=value, pantry=["chicken"]))
        assert query.dish == ""
        assert query.min_coverage == DEFAULT_MIN_COVERAGE


def test_a_dish_and_a_pantry_can_coexist():
    """"A chicken pot pie using what I have" is both."""
    query = build_query(payload(dish="pot pie", pantry=["chicken", "peas"]))

    assert query.dish == "pot pie"
    assert query.pantry == ("chicken", "peas")
