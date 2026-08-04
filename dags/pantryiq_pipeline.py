"""The batch pipeline, end to end: Bronze -> Silver -> entity resolution -> Gold, gated.

Every task calls the same `main()` a human would run by hand, so the DAG orchestrates the
documented interface rather than a second, parallel implementation that could drift from it.

**The DAG is strictly sequential, and that is a correctness requirement, not a style choice.**
Silver and Gold share one DuckDB file, and DuckDB takes an exclusive write lock — two tasks
writing at once do not interleave, they fail. Three things enforce it, because each covers a
gap the others cannot: `max_active_tasks = 1` (inside a DAG), the shared `duckdb_writer` **pool**
(across both DAGs), and an `flock` inside every writing `main()` (against a human running a
module in a terminal while the pipeline is mid-run).

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

from pathlib import Path

import pendulum
from airflow.sdk import dag, task

# ONE pool shared by both DAGs. `max_active_tasks` bounds concurrency inside a DAG and
# `max_active_runs` bounds runs of the same DAG — neither can stop `pantryiq_ingest` and
# `pantryiq_pipeline` running together, and DuckDB's exclusive write lock makes that a failure
# rather than a slowdown. Create it once:
#   uv run airflow pools set duckdb_writer 1 "serialises every warehouse writer"
WRITER_POOL = "duckdb_writer"

DEFAULT_ARGS = {
    "pool": WRITER_POOL,
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
    def build_recipe_meta():
        """Bronze recipes -> silver.recipe_meta: the titles the agent names recipes by."""
        from pantryiq.silver.recipe_meta import main

        main()

    @task
    def build_usda_categories():
        """Bronze categories -> silver.usda_food_categories: USDA's own 25 food groups.

        What `gold.recipe_tags` allows against. Without it the dietary tags fall back to matching
        substrings in product names, which cannot support a safety claim.
        """
        from pantryiq.silver.usda_categories import main

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

    @task.bash(cwd=str(Path(__file__).resolve().parents[1]))
    def build_gold_gated() -> str:
        """dbt build into STAGING: models and tests interleaved, so a failed test skips the rest.

        A non-zero exit fails the task, which stops `publish_gold` from running — that is what
        makes this an admission gate rather than only a propagation one. `cwd` is derived from
        __file__ rather than $AIRFLOW_HOME, which is unset by default and silently resolves to
        $HOME (dbt-duckdb then creates an empty warehouse and reports it healthy).
        """
        return "uv run dbt build --profiles-dir . --target staging"

    @task
    def publish_gold():
        """Swap staging into `gold` in one transaction and stamp the vintage.

        Only reached when the gate passed, so Gold moves all-at-once from a fully-tested build.
        """
        from pantryiq.gold.publish import export, publish

        publish()
        export()

    @task
    def check_distributions():
        """Advisory distribution checks over published Gold — reports, does not gate.

        Runs AFTER publish on purpose: it describes what was actually shipped. Blocking here
        would refuse a build for having moved, and movement is a finding to read.
        """
        from pantryiq.gold.expectations import main

        main()

    # Silver fan-in, then resolution, then Gold. Written as one chain because the DuckDB write
    # lock makes concurrency a failure mode rather than a speed-up.
    (
        parse_ingredient_lines()
        >> normalize_usda_foods()
        >> parse_usda_portions()
        >> extract_servings()
        >> build_recipe_meta()
        >> build_usda_categories()
        >> generate_candidates()
        >> fit_confidence_curve()
        >> fit_abstain_threshold()
        >> build_entity_map()
        >> convert_to_grams()
        >> report_gram_accuracy()
        >> build_gold_gated()
        >> publish_gold()
        >> check_distributions()
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

    @task
    def ingest_usda_categories():
        """A second ~410-request pass over the same ids, for `foodCategory`.

        Its own pull because the portions cache kept only `fdcId` and `foodPortions` and threw
        the rest of the full-format response away. Resumable in the same way.
        """
        from pantryiq.ingestion.categories import main

        main()

    (ingest_recipes() >> ingest_usda_foods() >> ingest_usda_portions()
     >> ingest_usda_categories())


pantryiq_pipeline()
pantryiq_ingest()
