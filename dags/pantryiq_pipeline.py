"""The batch pipeline, end to end: Bronze -> Silver -> entity resolution -> Gold, gated.

Every task calls the same `main()` a human would run by hand, so the DAG orchestrates the
documented interface rather than a second, parallel implementation that could drift from it.

**The DAG is strictly sequential, and that is a correctness requirement, not a style choice.**
Silver and Gold share one DuckDB file, and DuckDB takes an exclusive write lock — two tasks
writing at once do not interleave, they fail. `max_active_tasks = 1` states that in the one
place someone would look before adding parallelism.

**Idempotent by construction.** Every step does `CREATE OR REPLACE` or an Iceberg `overwrite`,
so a re-run from any point converges to the same tables. Nothing appends. The one guard worth
knowing: a 0-row ingest is *refused* rather than allowed to atomically wipe its Bronze table.

**The gate is `dbt build`, not `dbt run` then `dbt test`.** The plan's original shape had them
as separate tasks; that is wrong here, and §14 explains why — run-then-test rebuilds every Gold
table from bad data and reports the failure afterwards, while build interleaves tests with
models so a failure skips everything downstream. Splitting them back into two Airflow tasks
would reintroduce exactly the hole the gate exists to close.

**Ingestion is off the default path.** The RecipeNLG and USDA pulls are slow (the portions pull
is ~410 API requests) and their inputs are frozen — the gold labels and every §7-§14 metric are
pinned to that exact entity set. Re-pulling on a schedule would silently move the ground truth
under the measurements. `ingest_sources` is therefore a manually-triggered DAG of its own.

Run locally:  uv run airflow dags test pantryiq_pipeline
"""
from __future__ import annotations

import pendulum
from airflow.sdk import dag, task

DEFAULT_ARGS = {
    # Retries are for transient failures (a locked DB, a flaky FS), not for logic errors —
    # a failing quality gate re-runs to the same failure, which is the point of a gate.
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=1),
}


@dag(
    dag_id="pantryiq_pipeline",
    description="Silver -> entity resolution -> Gold, gated by dbt build",
    schedule=None,  # manual; the corpus is a frozen snapshot, not a stream
    start_date=pendulum.datetime(2026, 7, 1, tz="UTC"),
    catchup=False,
    max_active_tasks=1,  # DuckDB single-writer lock — see the module docstring
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["pantryiq", "silver", "gold"],
)
def pantryiq_pipeline():
    """Everything downstream of Bronze. Bronze itself is `pantryiq_ingest`."""

    @task
    def parse_ingredient_lines():
        """Bronze recipes -> silver.recipe_ingredient_lines + distinct_ingredient_strings."""
        from pantryiq.silver.ingredient_lines import main

        main()

    @task
    def normalize_usda_foods():
        """Bronze USDA -> silver.usda_foods, the canonical entity set."""
        from pantryiq.silver.usda_foods import main

        main()

    @task
    def parse_usda_portions():
        """Bronze portions -> silver.usda_portions, the gram weights."""
        from pantryiq.silver.usda_portions import main

        main()

    @task
    def extract_servings():
        """Bronze recipes -> silver.recipe_servings, for the ~15% that state one."""
        from pantryiq.silver.servings import main

        main()

    @task
    def generate_candidates():
        """Embed and block: silver.ingredient_candidates. The slow step (~1 min)."""
        from pantryiq.er.candidates import main

        main()

    @task
    def fit_confidence_curve():
        """Refit the isotonic curve on tune -> data/confidence_curve.json."""
        from pantryiq.er.resolve import main

        main()

    @task
    def fit_abstain_threshold():
        """Refit the no-match cut on tune -> data/abstain_threshold.json."""
        from pantryiq.er.abstain import main

        main()

    @task
    def build_entity_map():
        """silver.ingredient_entity_map + the stratified corpus headline."""
        from pantryiq.er.entity_map import main

        main()

    @task
    def convert_to_grams():
        """silver.recipe_ingredient_grams — the mass every Gold number scales."""
        from pantryiq.silver.grams import main

        main()

    @task
    def report_gram_accuracy():
        """3.3b — accuracy against published reference weights, not just coverage.

        Reporting only; it does not gate. The gate enforces what is *impossible*, while this
        measures what is *wrong*, and a metric that drifts is a finding to read rather than a
        reason to refuse to build.
        """
        from pantryiq.silver.gram_accuracy import main

        main()

    @task.bash
    def build_gold_gated() -> str:
        """dbt build: models and tests interleaved, so a failed test blocks promotion.

        A non-zero exit fails the task and therefore the run, which is the gate.
        """
        return "cd $AIRFLOW_HOME && uv run dbt build --profiles-dir ."

    # Silver fan-in, then resolution, then Gold. Written as one chain because the DuckDB write
    # lock makes concurrency a failure mode rather than a speed-up.
    (
        parse_ingredient_lines()
        >> normalize_usda_foods()
        >> parse_usda_portions()
        >> extract_servings()
        >> generate_candidates()
        >> fit_confidence_curve()
        >> fit_abstain_threshold()
        >> build_entity_map()
        >> convert_to_grams()
        >> report_gram_accuracy()
        >> build_gold_gated()
    )


@dag(
    dag_id="pantryiq_ingest",
    description="Bronze ingestion — manual only; re-pulling moves the frozen ground truth",
    schedule=None,
    start_date=pendulum.datetime(2026, 7, 1, tz="UTC"),
    catchup=False,
    max_active_tasks=1,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["pantryiq", "bronze"],
)
def pantryiq_ingest():
    """Deliberately separate from the main pipeline.

    These pulls are slow and, more importantly, their outputs are the frozen reference the 300
    gold labels and every §7-§14 metric are pinned to. Running them on a schedule would move the
    ground truth under the measurements without anyone deciding to.
    """

    @task
    def ingest_recipes():
        from pantryiq.ingestion.recipes import main

        main()

    @task
    def ingest_usda_foods():
        from pantryiq.ingestion.usda import main

        main()

    @task
    def ingest_usda_portions():
        """~410 API requests. Resumable: completed batches are cached, so a retry is cheap."""
        from pantryiq.ingestion.portions import main

        main()

    ingest_recipes() >> ingest_usda_foods() >> ingest_usda_portions()


pantryiq_pipeline()
pantryiq_ingest()
