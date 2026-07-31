"""The quality gate: the generic tests it is built from, and the bounds it enforces."""
from pathlib import Path

import pytest
import yaml

SCHEMA = Path("models/gold/_schema.yml")


def _tests_for(model_name):
    schema = yaml.safe_load(SCHEMA.read_text())
    model = next(m for m in schema["models"] if m["name"] == model_name)
    found = set()
    for column in model.get("columns", []):
        for test in column.get("tests", []):
            name = test if isinstance(test, str) else next(iter(test))
            found.add((column["name"], name))
    return found


def test_referential_integrity_is_enforced_on_resolved_ingredients():
    """A resolved ingredient pointing at an entity that does not exist would put fabricated
    nutrition into Gold with nothing to trace it back to. This is the one test the brief names
    explicitly (§6): "every resolved ingredient -> a real canonical entity"."""
    assert ("canonical_id", "relationships") in _tests_for("recipe_ingredients_resolved")


def test_the_trust_score_and_coverages_are_bounded_to_a_probability():
    """data_trust_score is surfaced in the product as a quality signal. A value outside [0,1]
    means the formula broke, and it would still render as a plausible-looking number."""
    bounded = _tests_for("recipe_nutrition")
    for column in ("data_trust_score", "nutrition_coverage", "cost_coverage"):
        assert (column, "accepted_range") in bounded, column


def test_energy_density_is_bounded_by_physics_not_by_plausibility():
    """910 kcal/100 g is above pure fat (~902) and below anything achievable by a mixture. A
    gate tuned to what looks *unusual* gets muted by real data — this corpus legitimately
    contains 40 lb of pork fat and 13 gallons of ice cream — and a muted gate is worse than
    none."""
    schema = yaml.safe_load(SCHEMA.read_text())
    for model_name in ("canonical_ingredients", "recipe_nutrition"):
        model = next(m for m in schema["models"] if m["name"] == model_name)
        column = next(c for c in model["columns"] if c["name"] == "kcal_per_100g")
        bounds = next(t["accepted_range"] for t in column["tests"]
                      if isinstance(t, dict) and "accepted_range" in t)
        assert bounds["max_value"] == 910, model_name


def test_grams_and_prices_must_be_strictly_positive():
    """Zero is the dangerous value, not negative: a 0 g ingredient or a $0 price still counts as
    covered while contributing nothing, which understates a recipe and looks like a measurement.
    `inclusive: false` is what makes 0 a failure rather than a boundary."""
    schema = yaml.safe_load(SCHEMA.read_text())
    for model_name, column_name in (("recipe_ingredients_resolved", "grams"),
                                    ("recipe_nutrition", "total_grams"),
                                    ("ingredient_costs", "usd_per_kg")):
        model = next(m for m in schema["models"] if m["name"] == model_name)
        column = next(c for c in model["columns"] if c["name"] == column_name)
        bounds = next(t["accepted_range"] for t in column["tests"]
                      if isinstance(t, dict) and "accepted_range" in t)
        assert bounds.get("inclusive") is False, f"{model_name}.{column_name} allows 0"


def test_the_gate_runs_through_dbt_build_not_run_then_test():
    """`dbt run` + `dbt test` reports failures AFTER every table has been rebuilt from bad data.
    `dbt build` interleaves them, so a failure skips everything downstream. That single word is
    the difference between a gate and an alarm."""
    # Recipe lines only — the Makefile's comments discuss `dbt run` to explain why it is not
    # used, and an assertion over raw text would fire on the explanation itself.
    recipes = [line.strip() for line in Path("Makefile").read_text().splitlines()
               if line.startswith("\t")]

    assert any("dbt build" in line for line in recipes)
    assert not any("dbt run " in line for line in recipes)


@pytest.mark.skipif(not Path("data/pantryiq.duckdb").exists(), reason="corpus not present")
def test_gold_currently_satisfies_the_bounds_the_gate_enforces():
    """The gate is only meaningful if real data passes it — a gate that fires on production data
    gets disabled, and then it protects nothing."""
    import duckdb

    con = duckdb.connect("data/pantryiq.duckdb", read_only=True)
    try:
        breaches = con.execute("""
            SELECT count(*) FROM gold.recipe_nutrition
            WHERE data_trust_score NOT BETWEEN 0 AND 1
               OR nutrition_coverage NOT BETWEEN 0 AND 1
               OR (kcal_per_100g IS NOT NULL AND kcal_per_100g > 910)
               OR (total_grams IS NOT NULL AND total_grams <= 0)
        """).fetchone()[0]
    finally:
        con.close()

    assert breaches == 0
