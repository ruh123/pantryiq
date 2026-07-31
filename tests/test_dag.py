"""The Airflow DAGs: they must import, and their shape encodes two correctness constraints."""
import os
from pathlib import Path

import pytest

DAGS = Path("dags")
pytest.importorskip("airflow", reason="orchestration group not installed (uv sync --group orchestration)")


@pytest.fixture(scope="module")
def dagbag():
    os.environ.setdefault("AIRFLOW_HOME", str(Path.cwd()))
    from airflow.dag_processing.dagbag import DagBag

    return DagBag(str(DAGS))


def test_the_dags_import_without_errors(dagbag):
    """A DAG that fails to import is invisible in the UI rather than loudly broken."""
    assert dagbag.import_errors == {}
    assert set(dagbag.dags) == {"pantryiq_pipeline", "pantryiq_ingest"}


def test_the_pipeline_is_strictly_sequential(dagbag):
    """Not a style choice — Silver and Gold share one DuckDB file and DuckDB takes an exclusive
    write lock, so two concurrent tasks do not interleave, they fail. This is the assertion that
    should stop someone 'optimising' the DAG with parallelism."""
    for dag in dagbag.dags.values():
        assert dag.max_active_tasks == 1, dag.dag_id
        assert dag.max_active_runs == 1, dag.dag_id


def test_gold_is_built_through_a_gate_not_a_bare_run(dagbag):
    """The plan's original shape had dbt_run and dbt_test as separate tasks. That reintroduces
    the exact hole the gate closes: run-then-test rebuilds every Gold table from bad data and
    reports the failure afterwards. The DAG must call `dbt build`, which interleaves them."""
    source = (DAGS / "pantryiq_pipeline.py").read_text()

    assert "dbt build" in source
    assert "dbt run" not in source.replace("`dbt run`", "")  # prose about it is fine


def test_ingestion_is_separate_from_the_pipeline(dagbag):
    """Bronze pulls are slow AND their outputs are the frozen reference the 300 gold labels and
    every §7-§14 metric are pinned to. Putting them in the scheduled path would move the ground
    truth under the measurements without anyone deciding to."""
    pipeline = dagbag.dags["pantryiq_pipeline"]
    ingest = dagbag.dags["pantryiq_ingest"]

    assert not any("ingest" in task.task_id for task in pipeline.tasks)
    assert all(task.task_id.startswith("ingest_") for task in ingest.tasks)
    assert pipeline.schedule is None and ingest.schedule is None   # manual only


def test_tasks_retry_transient_failures(dagbag):
    """A locked DB or a flaky filesystem should not fail a 15-second run. A failing quality gate
    re-runs to the same failure, which is the point of a gate — retries do not paper over it."""
    for dag in dagbag.dags.values():
        assert dag.default_args["retries"] >= 1, dag.dag_id
