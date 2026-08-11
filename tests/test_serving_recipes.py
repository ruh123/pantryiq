"""The cooking method — and the boundary that keeps it away from the model.

The load-bearing test in this file is `test_directions_never_reach_the_model`. Everything else is
parsing. If the method ever enters `AnswerContext`, every oven temperature and timing in it
becomes a value the guardrail will accept from the model, and "about 425 calories" starts passing
a check built to stop exactly that.
"""
import json

import duckdb
import pytest

from pantryiq.agent.context import AnswerContext, RecipeFact, build
from pantryiq.agent.retrieval import PantryQuery
from pantryiq.serving.recipes import _steps, directions_for
from test_context import make_candidate

POT_PIE = ["combine the first five ingredients and place in a 9 x 12-inch casserole dish.",
           "mix next four ingredients and pour over top of chicken mixture.",
           "do not stir.",
           "bake at 425° for 45 minutes to one hour until crust rises and browns."]


@pytest.fixture
def gold(tmp_path):
    """A Gold export with just the column under test."""
    # Not `gold.duckdb`: DuckDB names the catalog after the file, and a catalog called `gold`
    # collides with the schema of the same name ("Ambiguous reference to catalog or schema").
    db = tmp_path / "export.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE SCHEMA gold")
    con.execute("CREATE TABLE gold.recipe_meta (recipe_id VARCHAR, title VARCHAR, "
                "source_url VARCHAR, directions VARCHAR)")
    con.executemany("INSERT INTO gold.recipe_meta VALUES (?, ?, ?, ?)", [
        ("r1", "Chicken Pot Pie", "http://example.test/1", json.dumps(POT_PIE)),
        ("r2", "No Method", "http://example.test/2", None),
        ("r3", "Empty Method", "http://example.test/3", "[]"),
        ("r4", "Broken Method", "http://example.test/4", "not json at all"),
        ("r5", "Not A List", "http://example.test/5", '{"step": "one"}'),
    ])
    con.close()
    return db


def test_the_method_comes_back_as_ordered_steps(gold):
    result = directions_for(["r1"], db_path=gold)

    assert result["r1"] == tuple(POT_PIE)
    assert result["r1"][0].startswith("combine the first five")


def test_recipes_without_a_usable_method_are_absent_rather_than_empty(gold):
    """The caller renders "no method published" for a miss, so an empty tuple in the map would be
    a second way to say the same thing and a second branch to get wrong."""
    result = directions_for(["r1", "r2", "r3", "r4", "r5"], db_path=gold)

    assert set(result) == {"r1"}


def test_a_malformed_method_does_not_take_the_answer_down(gold):
    """One bad scrape must not stop a nutrition question being answered."""
    assert directions_for(["r4"], db_path=gold) == {}
    assert directions_for(["r5"], db_path=gold) == {}


def test_no_ids_means_no_query(gold):
    assert directions_for([], db_path=gold) == {}


def test_every_card_is_served_by_one_query(gold):
    """Eight connections would cost more than the retrieval that produced the recipes."""
    result = directions_for(["r1", "r2", "r3"], db_path=gold)

    assert result == {"r1": tuple(POT_PIE)}


@pytest.mark.parametrize(("raw", "expected"), [
    (None, ()),
    ("", ()),
    ("[]", ()),
    ('["  ", ""]', ()),                       # whitespace-only steps are not steps
    ('["one", "  two  "]', ("one", "two")),   # and real ones are trimmed
    ("[1, 2]", ("1", "2")),                   # non-strings are coerced, not dropped
])
def test_step_parsing(raw, expected):
    assert _steps(raw) == expected


def test_directions_never_reach_the_model(gold):
    """**The reason this module exists.**

    A method carrying "bake at 425 for 45 minutes" would put 425 and 45 into the set of values the
    model may state. §16.3 measures 38% of injected fabrications already colliding with a real
    value; licensing every oven temperature in the corpus would widen that for nothing, since the
    answer never discusses the method.
    """
    context = AnswerContext(question="q", query=PantryQuery(pantry=("chicken",)),
                            recipes=(RecipeFact.from_candidate(make_candidate(recipe_id="r1")),))
    method = directions_for(["r1"], db_path=gold)["r1"]

    quotable = context.numbers()

    assert 425.0 not in quotable, "an oven temperature became a quotable nutrition value"
    assert 45.0 not in quotable
    assert 12.0 not in quotable
    # And the method's text is nowhere in what the model is shown.
    prompt = context.to_prompt()
    for step in method:
        assert step not in prompt
    assert "bake at" not in prompt.lower()


def test_the_context_builder_still_ignores_the_new_column():
    """`RecipeFact` is built from `Candidate`, and `retrieve` selects an explicit column list —
    so adding `directions` to Gold must not have widened either. Guards against a future
    `SELECT *`."""
    assert not hasattr(make_candidate(), "directions")
    assert "directions" not in RecipeFact.__dataclass_fields__


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/pantryiq_gold.duckdb").exists(),
    reason="export not present",
)
def test_the_real_export_carries_a_method_for_the_recipe_that_prompted_this():
    """Chicken Pot Pie is the example that surfaced the gap: the app listed a title and calories
    and could not show how to cook it."""
    context = build("chicken", PantryQuery(pantry=("chicken",), limit=8))
    methods = directions_for([recipe.recipe_id for recipe in context.recipes])

    assert methods, "no method found for any retrieved recipe"
    assert all(isinstance(steps, tuple) and steps for steps in methods.values())


def test_an_unreachable_database_yields_no_method_rather_than_raising(tmp_path):
    """This is the corpus-free property, and it broke in CI before it was pinned.

    `render_ledger` calls this on every answer, so it is a second path to the database that no
    façade stub covers — on a checkout without the Gold export it raised an IOException from
    inside the page. `startup_problem()` already refuses to run the app without that file, so a
    failure here is transient, and losing a garnish must not lose an answer that passed the
    guardrail.
    """
    assert directions_for(["r1"], db_path=tmp_path / "does-not-exist.duckdb") == {}


def test_a_database_without_the_table_also_degrades(tmp_path):
    """An older export, or one built before `directions` was promoted to Gold."""
    db = tmp_path / "old-export.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE SCHEMA gold")
    con.execute("CREATE TABLE gold.recipe_meta (recipe_id VARCHAR, title VARCHAR)")
    con.close()

    assert directions_for(["r1"], db_path=db) == {}
