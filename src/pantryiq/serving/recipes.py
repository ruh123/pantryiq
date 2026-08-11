"""The cooking method — read for the screen, and deliberately not for the model.

This is a separate module from `answer.py`, and the separation is the whole design.

`AnswerContext` is the guardrail's source of truth: every value in it is a number the model is
permitted to state, and anything else in its response is rejected as a fabrication. A method
reading *"bake at 425 for 45 minutes to one hour"* carries `425`, `45`, `9` and `12`. Putting
those in the ledger would license "about 425 calories" to pass a check built to stop precisely
that — and §16.3 already measures 38% of injected fabrications colliding with a legitimate value,
so widening the permitted set costs real catch rate. The model has no reason to discuss the method
anyway; it answers about nutrition, cost and what you have in the cupboard.

So the directions travel on their own path: retrieval returns facts, this returns text, the page
shows both, and the two never meet. Nothing here is ever passed to `generate` or `guardrail`.

**It is untrusted scraped text** and reaches a browser, so callers escape it — see `app.py`.

Worth showing for a second reason. `nutrition_coverage = 1.0` means every line we *have* was
weighed, not that the list is complete, and this corpus ships truncated recipes: "Pickled Bologna"
lists vinegar, sugar, salt and pickling spice, and no bologna. The method is the one field that
can expose that, which turns a caveat into something a reader can see for themselves.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

from pantryiq.agent.retrieval import DEFAULT_DB


def _steps(raw: str | None) -> tuple[str, ...]:
    """The stored JSON array as steps, tolerating anything that is not one.

    RecipeNLG stores the method as a JSON-encoded list of strings. A row that is null, empty, or
    not a list yields no steps rather than raising: a missing method is a gap to render, and the
    page must not fail to answer a nutrition question because one scrape was malformed.
    """
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(step).strip() for step in parsed if str(step).strip())


def directions_for(recipe_ids, db_path: Path | str = DEFAULT_DB) -> dict[str, tuple[str, ...]]:
    """Method steps per recipe id. Ids with no usable method are absent from the result.

    One query for the whole answer rather than one per card — eight round trips to open and close
    a DuckDB connection would cost more than the retrieval that produced the recipes.
    """
    ids = [str(recipe_id) for recipe_id in recipe_ids]
    if not ids:
        return {}

    placeholders = ", ".join("?" for _ in ids)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            f"SELECT recipe_id, directions FROM gold.recipe_meta "  # noqa: S608 - ids are bound
            f"WHERE recipe_id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        con.close()

    found = {recipe_id: _steps(raw) for recipe_id, raw in rows}
    return {recipe_id: steps for recipe_id, steps in found.items() if steps}


def main() -> None:
    from pantryiq.agent.context import build
    from pantryiq.agent.retrieval import PantryQuery

    context = build("chicken pot pie", PantryQuery(pantry=("chicken",), limit=3))
    methods = directions_for([recipe.recipe_id for recipe in context.recipes])

    for recipe in context.recipes:
        steps = methods.get(recipe.recipe_id, ())
        print(f"\n{recipe.title or recipe.recipe_id} — {len(steps)} steps")
        for index, step in enumerate(steps, 1):
            print(f"  {index}. {step}")

    print("\n  None of this reached the model: the context holds "
          f"{len(context.numbers())} quotable values and not one of them came from a method.")


if __name__ == "__main__":
    main()
