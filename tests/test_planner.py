"""The week planner — constraint solving that must happen in code, not in a model."""


from pantryiq.agent.context import RecipeFact
from pantryiq.agent.guardrail import check
from pantryiq.agent.planner import Plan, build_plan, candidates, plan_context
from pantryiq.agent.retrieval import Candidate
from test_context import make_candidate


def priced(recipe_id: str, kcal: float, cost: float, **extra) -> Candidate:
    return make_candidate(recipe_id=recipe_id, title=recipe_id, total_kcal=kcal,
                          cost_total_usd=cost, cost_coverage=1.0, nutrition_coverage=1.0,
                          **extra)


def a_plan(*recipes, budget=50.0, ceiling=2000.0) -> Plan:
    return Plan(recipes=recipes, budget_usd=budget, kcal_per_day=ceiling, days=7)


def test_a_plan_that_breaks_its_budget_reports_it():
    """`violations` is what makes 'verified' true rather than believed. The selection loop is
    the thing being checked, so the check cannot depend on the loop being right."""
    over = a_plan(priced("a", 500, 30.0), priced("b", 500, 30.0))

    assert any("exceeds" in problem for problem in over.violations())


def test_a_day_over_the_calorie_ceiling_reports_it():
    over = a_plan(priced("a", 2500, 1.0))

    assert any("daily ceiling" in problem for problem in over.violations())


def test_the_same_recipe_twice_reports_it():
    """A week of one dish satisfies every arithmetic constraint and is not a plan."""
    repeated = a_plan(priced("a", 500, 1.0), priced("a", 500, 1.0))

    assert any("twice" in problem for problem in repeated.violations())


def test_a_valid_plan_has_no_violations():
    assert a_plan(priced("a", 1900, 6.0), priced("b", 1800, 4.0)).violations() == []


def test_the_verified_totals_are_quotable_but_a_model_derived_sum_is_not():
    """The whole reason `computed` exists. Generation is forbidden from summing — a model adding
    seven costs is doing arithmetic nobody checked — but a total the planner computed and
    verified is a fact. Before this channel existed the narration refused to describe its own
    plan, saying it could not total costs across days."""
    plan = a_plan(priced("a", 1900, 6.0), priced("b", 1800, 4.0))
    context = plan_context(plan)

    assert check("The week costs $10.00 of your $50 budget.", context).passed
    assert not check("The week costs $11.00.", context).passed


def test_the_narration_context_carries_the_budget_and_the_remainder():
    plan = a_plan(priced("a", 1900, 6.0), priced("b", 1800, 4.0))

    labels = dict(plan_context(plan).computed)

    assert labels["total_cost_usd"] == 10.0
    assert labels["budget_remaining_usd"] == 40.0
    assert labels["days_planned"] == 2


def test_a_recipe_with_no_serving_count_still_reports_no_per_serving_figure():
    """The planner runs on totals precisely because only 14 recipes have both a serving count
    and complete cost. It must never manufacture the denominator it declined to use."""
    fact = RecipeFact.from_candidate(priced("a", 1900, 6.0, servings=None))

    assert fact.kcal_per_serving is None
    assert "<kcal_per_serving>unknown</kcal_per_serving>" in plan_context(
        a_plan(priced("a", 1900, 6.0, servings=None))).to_prompt()


def test_candidates_all_have_complete_nutrition_and_complete_cost(fixture_warehouse):
    """A budget is a statement about money, and a cost covering 60% of a recipe is a floor, not
    a price. Planning against floors produces a week that looks affordable and is not."""
    pool = candidates(2000.0, fixture_warehouse)

    assert pool, "no candidates at all"
    assert all(recipe.nutrition_coverage == 1.0 for recipe in pool)
    assert all(recipe.cost_coverage == 1.0 for recipe in pool)
    assert all(recipe.total_kcal <= 2000.0 for recipe in pool)


def test_a_real_plan_holds_its_constraints(fixture_warehouse):
    plan = build_plan(budget_usd=50.0, kcal_per_day=2000.0, days=7,
                      db_path=fixture_warehouse)

    assert plan.violations() == []
    assert plan.recipes
    assert plan.total_cost <= 50.0


def test_an_impossible_budget_yields_a_short_week_not_a_repeated_one(fixture_warehouse):
    """A short honest plan beats a padded one. The loop stops rather than reusing a recipe."""
    plan = build_plan(budget_usd=0.02, kcal_per_day=2000.0, days=7, db_path=fixture_warehouse)

    assert plan.violations() == []
    assert len(plan.recipes) < 7
    assert plan.total_cost <= 0.02


def test_a_plan_costing_exactly_the_budget_is_legal():
    """`>` vs `>=` survived on all four bounds because every fixture sat strictly inside them.
    A week that spends the budget exactly has not exceeded it."""
    exact = a_plan(priced("a", 1000, 25.0), priced("b", 1000, 25.0), budget=50.0)

    assert exact.violations() == []


def test_a_recipe_at_exactly_the_daily_ceiling_is_legal():
    assert a_plan(priced("a", 2000, 5.0), ceiling=2000.0).violations() == []


def test_the_candidate_pool_includes_a_recipe_at_exactly_the_ceiling(fixture_warehouse):
    """The pool filter is `total_kcal <= ?`; `<` survived twice — first because nothing sat on
    the boundary, then because asserting the pool is merely non-empty still passes when the one
    recipe ON the boundary is the only thing dropped. The boundary recipe must be PRESENT."""
    pool = candidates(2000.0, fixture_warehouse)
    heaviest = max(recipe.total_kcal for recipe in pool)

    at_ceiling = candidates(heaviest, fixture_warehouse)

    assert all(recipe.total_kcal <= 2000.0 for recipe in pool)
    assert heaviest in [recipe.total_kcal for recipe in at_ceiling], \
        "the recipe at exactly the ceiling was excluded"


def test_the_greedy_loop_can_spend_the_last_cent_of_the_budget(fixture_warehouse):
    """`cost <= remaining` -> `<` inside `build_plan` survived: the boundary tests all built a
    `Plan` directly and checked `violations()`, so the SELECTION loop's own comparison — the code
    that decides what goes in — was never exercised on the boundary."""
    pool = candidates(2000.0, fixture_warehouse)
    cheapest = min(recipe.cost_total_usd for recipe in pool)

    plan = build_plan(budget_usd=cheapest, kcal_per_day=2000.0, days=1,
                      db_path=fixture_warehouse)

    assert len(plan.recipes) == 1, "a recipe costing exactly the budget was not affordable"
    assert plan.violations() == []
