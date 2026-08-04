"""The week planner — constraint solving in code, with Claude narrating the result.

"A week under $50 and 2,000 calories a day" is arithmetic and selection, and the brief is
explicit that the model does neither. The plan is chosen here, verified here, and only then
described. If the narration and the plan ever disagree, the guardrail rejects the narration.

**It plans on recipe TOTALS, and says so.** Only 118 of 15,000 recipes carry a per-serving
figure, and just 14 of those also have complete cost — so a per-person-per-day plan would rest on
a serving count invented for the other 97 recipes. §3.4 refused to publish exactly that quotient.
The assumption travels with the plan: one recipe is one day's food, and how many people that
feeds is the cook's judgement, not ours.

**Candidates need complete nutrition AND complete cost.** 111 recipes qualify. A budget is a
statement about money, and a cost figure covering 60% of a recipe's mass is not one — it is a
floor. Planning against floors would produce a week that looks affordable and is not.

Selection is greedy on calories descending, subject to the daily ceiling and the remaining
budget, so a week is substantial meals rather than the seven cheapest sauces. Cheap to explain,
which matters more here than optimality: the user is entitled to know why these seven.

Run:  uv run python -m pantryiq.agent.planner
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from pantryiq.agent.context import AnswerContext, RecipeFact
from pantryiq.agent.retrieval import DEFAULT_DB, Candidate, PantryQuery

DEFAULT_BUDGET_USD = 50.0
DEFAULT_KCAL_PER_DAY = 2000.0
DEFAULT_DAYS = 7


@dataclass(frozen=True)
class Plan:
    """A chosen week, plus the constraints it was chosen under."""

    recipes: tuple[Candidate, ...]
    budget_usd: float
    kcal_per_day: float
    days: int

    @property
    def total_cost(self) -> float:
        return sum(recipe.cost_total_usd for recipe in self.recipes)

    @property
    def total_kcal(self) -> float:
        return sum(recipe.total_kcal for recipe in self.recipes)

    def violations(self) -> list[str]:
        """Every way this plan fails its own constraints. Empty means it holds.

        Checked rather than assumed: the selection loop is what enforces these, and a plan that
        is only correct because its author believed the loop was correct is not verified.
        """
        problems = []
        if self.total_cost > self.budget_usd:
            problems.append(f"cost ${self.total_cost:.2f} exceeds ${self.budget_usd:.2f}")
        for recipe in self.recipes:
            if recipe.total_kcal > self.kcal_per_day:
                problems.append(f"{recipe.title} is {recipe.total_kcal:,.0f} kcal, "
                                f"over the {self.kcal_per_day:,.0f} daily ceiling")
        if len({recipe.recipe_id for recipe in self.recipes}) != len(self.recipes):
            problems.append("the same recipe appears twice")
        return problems


def candidates(kcal_ceiling: float, db_path: Path | str = DEFAULT_DB) -> list[Candidate]:
    """Recipes whose calories AND cost are both fully measured, under the daily ceiling."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT n.recipe_id, m.title,
                   n.ingredient_count, n.counted_ingredients,
                   n.total_kcal, n.kcal_per_100g, n.total_grams,
                   n.total_protein_g, n.total_fat_g, n.total_carb_g,
                   n.servings, n.kcal_per_serving,
                   n.cost_total_usd, n.cost_coverage,
                   n.nutrition_coverage, n.data_trust_score,
                   coalesce(g.tags, []) AS tags
            FROM gold.recipe_nutrition n
            LEFT JOIN gold.recipe_meta m USING (recipe_id)
            LEFT JOIN (SELECT recipe_id, list(tag) AS tags FROM gold.recipe_tags GROUP BY 1) g
                   USING (recipe_id)
            WHERE n.nutrition_coverage = 1.0 AND n.cost_coverage = 1.0
              AND n.total_kcal IS NOT NULL AND n.cost_total_usd IS NOT NULL
              AND n.total_kcal <= ?
            ORDER BY n.total_kcal DESC, n.recipe_id
        """, [kcal_ceiling]).fetchall()
    finally:
        con.close()

    return [Candidate(recipe_id=row[0], title=row[1], matched=(), missing=(),
                      ingredient_count=row[2], counted_ingredients=row[3],
                      total_kcal=row[4], kcal_per_100g=row[5], total_grams=row[6],
                      protein_g=row[7], fat_g=row[8], carb_g=row[9],
                      servings=row[10], kcal_per_serving=row[11],
                      cost_total_usd=row[12], cost_coverage=row[13],
                      nutrition_coverage=row[14], data_trust_score=row[15],
                      tags=tuple(sorted(row[16])))
            for row in rows]


def build_plan(budget_usd: float = DEFAULT_BUDGET_USD,
               kcal_per_day: float = DEFAULT_KCAL_PER_DAY,
               days: int = DEFAULT_DAYS, db_path: Path | str = DEFAULT_DB) -> Plan:
    """Greedy: the most substantial meal that still fits the money left, day after day."""
    remaining = budget_usd
    chosen: list[Candidate] = []
    pool = candidates(kcal_per_day, db_path)

    for _ in range(days):
        affordable = [recipe for recipe in pool
                      if recipe.cost_total_usd <= remaining
                      and recipe.recipe_id not in {c.recipe_id for c in chosen}]
        if not affordable:
            break  # a short week is an honest answer; a repeated recipe is not
        pick = affordable[0]  # pool is sorted by calories descending
        chosen.append(pick)
        remaining -= pick.cost_total_usd

    return Plan(recipes=tuple(chosen), budget_usd=budget_usd,
                kcal_per_day=kcal_per_day, days=days)


def plan_context(plan: Plan) -> AnswerContext:
    """The plan as an `AnswerContext`, so the narration is guardrailed like any other answer."""
    question = (f"Plan {plan.days} days of meals under ${plan.budget_usd:.0f} total, "
                f"with no day over {plan.kcal_per_day:,.0f} calories.")
    return AnswerContext(
        question=question,
        query=PantryQuery(max_kcal=plan.kcal_per_day, max_cost_usd=plan.budget_usd,
                          min_coverage=1.0, limit=plan.days),
        recipes=tuple(RecipeFact.from_candidate(recipe) for recipe in plan.recipes),
        # Computed and verified by `Plan.violations()` before the model sees any of it.
        computed=(("days_planned", len(plan.recipes)),
                  ("total_cost_usd", round(plan.total_cost, 2)),
                  ("budget_usd", plan.budget_usd),
                  ("budget_remaining_usd", round(plan.budget_usd - plan.total_cost, 2)),
                  ("total_kcal_for_the_week", round(plan.total_kcal)),
                  ("daily_kcal_ceiling", plan.kcal_per_day)))


def main() -> None:
    from pantryiq.agent.claude import get_client
    from pantryiq.agent.generate import answer
    from pantryiq.agent.guardrail import check
    from pantryiq.agent.log import record

    plan = build_plan()
    problems = plan.violations()

    print(f"{len(plan.recipes)} days planned, budget ${plan.budget_usd:.2f}, "
          f"ceiling {plan.kcal_per_day:,.0f} kcal/day")
    print(f"  candidates with complete nutrition AND cost: "
          f"{len(candidates(plan.kcal_per_day)):,}\n")
    for index, recipe in enumerate(plan.recipes, 1):
        print(f"  day {index}  {str(recipe.title)[:38]:40} "
              f"{recipe.total_kcal:6,.0f} kcal  ${recipe.cost_total_usd:5.2f}")
    print(f"\n  total ${plan.total_cost:.2f} of ${plan.budget_usd:.2f}  |  "
          f"{plan.total_kcal:,.0f} kcal over {len(plan.recipes)} days")
    print(f"  constraints verified in code: {'HOLD' if not problems else problems}")

    context = plan_context(plan)
    text = answer(context, get_client())
    verdict = check(text, context)
    record(context, text, verdict, regenerated=False)

    print(f"\n  narration guardrail: {'PASS' if verdict.passed else 'FAIL'} on "
          f"{verdict.checked} numeric claims"
          f"{'' if verdict.passed else '  ' + str(list(verdict.unsupported))}\n")
    print(text)
    print("\n  The plan was chosen and checked in code. Claude described it and nothing more —")
    print("  every number it wrote had to already be in the plan.")


if __name__ == "__main__":
    main()
