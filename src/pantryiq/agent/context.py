"""Context assembly — the contract between retrieval and the guardrail.

Generation sees this object and nothing else. That is what makes "the agent can only speak from
verified data" a checkable property rather than a claim: every fact available to the model is
enumerable here, so 4.5 can extract the numbers from a response and require each one to appear
in this object or be a permitted derivation of values in it.

**Rounding happens here, once, before either consumer sees a number.** The prompt shows
`3,470 kcal`; if the context held `3470.2481` the guardrail would then have to decide whether the
model had quoted a real value or invented one, and every rounded figure becomes a candidate false
positive. Building the context from display-rounded values instead makes the two agree by
construction — what the model is shown *is* what the guardrail checks.

**Every nutrition figure travels with its coverage.** A recipe at 0.8 coverage has a fifth of its
ingredients unweighed, and `total_kcal` is a sum over the four fifths we could weigh. The number
is not wrong, but it is not the recipe's energy either, and an answer that omits that is making a
stronger claim than the warehouse supports. Coverage is a required field, never optional.

**Untrusted text is tagged, not interpolated.** Recipe titles come from a public scrape and reach
a Claude prompt verbatim; `to_prompt` puts them inside `<recipe>` elements so a title reading
"ignore previous instructions" arrives as data. This mirrors `er/ensemble.py`.

**Division is not a permitted derivation, and this context is why.** A recipe can carry
`servings: 8` and `kcal_per_serving: unknown` at the same time — Phase 3 publishes per-serving
figures only at complete coverage, because dividing a partial total by a real denominator yields
a confident number for an energy value we know is short. The context states both facts, so a
model will be tempted to divide, and 3,470 / 8 = 433.75 is exactly the figure §3.4 refused to
publish. The guardrail must therefore admit sums, counts and ordinals but **not quotients**:
433.75 is absent from `numbers()`, and that is the mechanism that stops it.

Run:  uv run python -m pantryiq.agent.context
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape

from pantryiq.agent.retrieval import Candidate, PantryQuery, retrieve

# How each quantity is rounded for display, and therefore how it is stored. Money gets 2 places;
# energy and mass are quoted whole because a tenth of a kcal is noise against 22% gram error.
ROUNDING = {
    "total_kcal": 0, "kcal_per_100g": 0, "total_grams": 0, "kcal_per_serving": 0,
    "protein_g": 1, "fat_g": 1, "carb_g": 1,
    "cost_total_usd": 2, "cost_coverage": 2, "nutrition_coverage": 2, "data_trust_score": 2,
}


def _round(field: str, value: float | None) -> float | None:
    """Round to the field's display precision, keeping `None` distinct from zero."""
    if value is None:
        return None
    places = ROUNDING.get(field, 2)
    rounded = round(float(value), places)
    return int(rounded) if places == 0 else rounded


@dataclass(frozen=True)
class RecipeFact:
    """One recipe, reduced to exactly the facts an answer may quote."""

    recipe_id: str
    title: str | None
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    ingredient_count: int
    counted_ingredients: int
    nutrition_coverage: float
    data_trust_score: float
    total_kcal: float | None = None
    kcal_per_100g: float | None = None
    total_grams: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    carb_g: float | None = None
    servings: int | None = None
    kcal_per_serving: float | None = None
    cost_total_usd: float | None = None
    cost_coverage: float | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> RecipeFact:
        return cls(
            recipe_id=candidate.recipe_id,
            title=candidate.title,
            matched=candidate.matched,
            missing=candidate.missing,
            ingredient_count=candidate.ingredient_count,
            counted_ingredients=candidate.counted_ingredients,
            nutrition_coverage=_round("nutrition_coverage", candidate.nutrition_coverage),
            data_trust_score=_round("data_trust_score", candidate.data_trust_score),
            total_kcal=_round("total_kcal", candidate.total_kcal),
            kcal_per_100g=_round("kcal_per_100g", candidate.kcal_per_100g),
            total_grams=_round("total_grams", candidate.total_grams),
            protein_g=_round("protein_g", candidate.protein_g),
            fat_g=_round("fat_g", candidate.fat_g),
            carb_g=_round("carb_g", candidate.carb_g),
            servings=candidate.servings,
            kcal_per_serving=_round("kcal_per_serving", candidate.kcal_per_serving),
            cost_total_usd=_round("cost_total_usd", candidate.cost_total_usd),
            cost_coverage=_round("cost_coverage", candidate.cost_coverage),
            tags=candidate.tags,
        )

    def numbers(self) -> set[float]:
        """Every numeric value this recipe entitles an answer to state.

        Counts are included alongside measurements: an answer saying "uses 3 of your 4
        ingredients" is quoting `len(matched)` and `len(matched) + len(missing)`, and a guardrail
        that only knew about nutrition would reject it.
        """
        values = {float(self.ingredient_count), float(self.counted_ingredients),
                  float(len(self.matched)), float(len(self.missing)),
                  float(self.nutrition_coverage), float(self.data_trust_score),
                  # Coverage is quoted as a percentage at least as often as a fraction.
                  round(self.nutrition_coverage * 100, 1)}
        for field in ("total_kcal", "kcal_per_100g", "total_grams", "protein_g", "fat_g",
                      "carb_g", "servings", "kcal_per_serving", "cost_total_usd"):
            value = getattr(self, field)
            if value is not None:
                values.add(float(value))
        if self.cost_coverage is not None:
            values.add(float(self.cost_coverage))
            values.add(round(self.cost_coverage * 100, 1))
        return values


@dataclass(frozen=True)
class AnswerContext:
    """Everything generation is allowed to know, and the guardrail's source of truth."""

    question: str
    query: PantryQuery
    recipes: tuple[RecipeFact, ...]
    # Figures already computed in code, as (label, value) pairs — the week planner's totals, for
    # instance. Generation is forbidden from summing, which is right: a model adding seven costs
    # is doing arithmetic nobody checked. But a total the *planner* computed and verified is a
    # fact like any other, and without a channel for it the narration has to refuse to describe
    # its own plan. It did exactly that before this field existed.
    computed: tuple[tuple[str, float], ...] = ()
    # Names whose digits are not quantities — a dbt test's `unique_id`, a column called
    # `kcal_per_100g`. Recipe ids are handled automatically from `recipes`; this is for
    # everything else. Quoting an identifier is quoting the context, so its digits must not be
    # read as claims: `kcal_per_100g` was rejected as a fabricated "100".
    identifiers: tuple[str, ...] = ()

    def numbers(self) -> set[float]:
        """The union of every quotable value, plus the constraints the user themselves stated.

        The user's own numbers belong here: an answer to "under 500 calories" that says "all of
        these are under 500" is restating the question, not inventing a measurement.
        """
        values = {float(len(self.recipes))}
        for constraint in (self.query.max_kcal, self.query.min_kcal, self.query.max_cost_usd):
            if constraint is not None:
                values.add(float(constraint))
        for _, value in self.computed:
            values.add(float(value))
        for recipe in self.recipes:
            values |= recipe.numbers()
        return values

    def to_dict(self) -> dict:
        """JSON-round-trippable, for the query log and for tests."""
        return {
            "question": self.question,
            "query": {
                "pantry": list(self.query.pantry), "exclude": list(self.query.exclude),
                "max_kcal": self.query.max_kcal, "min_kcal": self.query.min_kcal,
                "kcal_basis": self.query.kcal_basis,
                "tags": list(self.query.tags), "max_cost_usd": self.query.max_cost_usd,
                "min_coverage": self.query.min_coverage, "limit": self.query.limit,
            },
            "recipes": [
                {**recipe.__dict__,
                 "matched": list(recipe.matched), "missing": list(recipe.missing),
                 "tags": list(recipe.tags)}
                for recipe in self.recipes
            ],
            "computed": [list(pair) for pair in self.computed],
        }

    def to_prompt(self) -> str:
        """The context as Claude sees it: one `<recipe>` element each, untrusted fields escaped.

        Absent values are rendered as `unknown` rather than omitted. A missing line invites the
        model to supply the figure from its own knowledge of what a meat loaf usually costs;
        `unknown` names the gap as a gap.
        """
        blocks = []
        for recipe in self.recipes:
            def show(value, unit: str = "", *, item=recipe) -> str:
                return "unknown" if value is None else f"{value:,}{unit}"

            blocks.append("\n".join([
                "<recipe>",
                f"  <title>{escape(recipe.title or 'untitled')}</title>",
                f"  <id>{recipe.recipe_id}</id>",
                f"  <uses_from_your_pantry>{', '.join(recipe.matched) or 'none'}"
                "</uses_from_your_pantry>",
                f"  <you_are_missing>{', '.join(recipe.missing) or 'nothing'}</you_are_missing>",
                f"  <ingredients_total>{recipe.ingredient_count}</ingredients_total>",
                f"  <nutrition_coverage>{recipe.nutrition_coverage} "
                f"({recipe.counted_ingredients} of {recipe.ingredient_count} ingredients weighed)"
                "</nutrition_coverage>",
                f"  <total_kcal>{show(recipe.total_kcal)}</total_kcal>",
                f"  <kcal_per_100g>{show(recipe.kcal_per_100g)}</kcal_per_100g>",
                f"  <total_grams>{show(recipe.total_grams)}</total_grams>",
                f"  <protein_g>{show(recipe.protein_g)}</protein_g>",
                f"  <fat_g>{show(recipe.fat_g)}</fat_g>",
                f"  <carb_g>{show(recipe.carb_g)}</carb_g>",
                f"  <servings>{show(recipe.servings)}</servings>",
                f"  <kcal_per_serving>{show(recipe.kcal_per_serving)}</kcal_per_serving>",
                f"  <cost_total_usd>{show(recipe.cost_total_usd)}</cost_total_usd>",
                f"  <cost_coverage>{show(recipe.cost_coverage)}</cost_coverage>",
                f"  <data_trust_score>{recipe.data_trust_score}</data_trust_score>",
                f"  <dietary_tags>{', '.join(recipe.tags) or 'none'}</dietary_tags>",
                "</recipe>",
            ]))
        body = "\n".join(blocks) or "<no_recipes_found/>"
        if self.computed:
            totals = "\n".join(f"  <{label}>{value:,}</{label}>" for label, value in self.computed)
            body += f"\n<already_computed>\n{totals}\n</already_computed>"
        return body


def build(question: str, query: PantryQuery, db_path=None) -> AnswerContext:
    """Retrieve for `query` and package the result. The only entry point generation needs."""
    candidates = retrieve(query) if db_path is None else retrieve(query, db_path)
    return AnswerContext(question=question, query=query,
                         recipes=tuple(RecipeFact.from_candidate(c) for c in candidates))


def main() -> None:
    query = PantryQuery(pantry=("chicken", "rice", "onion"), limit=2)
    context = build("What can I make with chicken, rice and onion?", query)

    print(context.to_prompt())
    print(f"\n  quotable values: {len(context.numbers())}")
    print(f"  {sorted(context.numbers())}")
    print("\n  Every number an answer may state is in that set. Anything else is a fabrication,")
    print("  and 4.5 rejects the response rather than shipping it.")

    assert json.loads(json.dumps(context.to_dict()))["recipes"][0]["recipe_id"]
    print("\n  round-trips to JSON: ok")


if __name__ == "__main__":
    main()
