"""Shared fixtures — chiefly a synthetic warehouse, so the safety rules run without the corpus.

Before this, 22 of the suite's tests were gated on `data/pantryiq_gold.duckdb` existing, so they
ran on a developer machine and nowhere else. Nine of them were the dietary-tag tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.warehouse import build_gold, build_silver  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixture_warehouse(tmp_path_factory) -> Path:
    """A synthetic Silver with the real Gold models built on top, once per session.

    Session-scoped because it shells out to `dbt build`, which costs a couple of seconds. The
    gold tables land in `gold_staging` — the same schema the production gate builds into.
    """
    path = tmp_path_factory.mktemp("warehouse") / "fixture.duckdb"
    build_silver(path)
    build_gold(path, PROJECT_ROOT)

    # Alias `gold_staging` to `gold` so a test reads `gold.recipe_tags` whether it is pointed at
    # the fixture or the real export, and the same assertions can run against either.
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS gold")
        for (name,) in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'gold_staging'"
        ).fetchall():
            con.execute(f"CREATE OR REPLACE VIEW gold.{name} AS SELECT * FROM gold_staging.{name}")
    finally:
        con.close()
    return path


@pytest.fixture(scope="session")
def fixture_gold(fixture_warehouse):
    """A read-only connection to the synthetic Gold, with `gold_staging` aliased to `gold`.

    The alias means a test reads `gold.recipe_tags` whether it is pointed at the fixture or the
    real export, so the same assertions can run against either.
    """
    con = duckdb.connect(str(fixture_warehouse), read_only=True)
    yield con
    con.close()
