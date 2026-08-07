"""Where the serving layer looks for its two DuckDB files.

Every path in `pantryiq` is relative to the process CWD, which is fine for `uv run` from the repo
root and wrong everywhere else. A container needs to point the reader at the Gold file it shipped
and the writer at a mount that survives a restart, and before Phase 5 neither was overridable —
only dbt honoured an env var (`PANTRYIQ_DB`, in `profiles.yml`).

Both constants are resolved at **import** time, so the assertion has to be made in a fresh
interpreter. Reloading the module in-process would rebind `Candidate` and `PantryQuery` to new
classes while `context.py` still held the old ones, leaving the rest of the suite running against
a half-swapped module.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolved(module: str, **env) -> str:
    """`DEFAULT_DB` as a fresh interpreter sees it, with `env` applied."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"from pantryiq.agent.{module} import DEFAULT_DB; print(DEFAULT_DB)"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_gold_reader_defaults_to_the_published_export():
    """The default is the artifact `gold/publish.py:export()` writes, and it must not move —
    every CLI entry point and every skipif in this suite names `data/pantryiq_gold.duckdb`."""
    assert resolved("retrieval", PANTRYIQ_GOLD_DB="") == "data/pantryiq_gold.duckdb"


def test_the_log_writer_defaults_to_its_own_file():
    """Deliberately not inside Gold: `publish()` swaps Gold wholesale in a transaction and the
    serving copy is read-only, so a write-heavy table cannot live there. See `log.py`."""
    assert resolved("log", PANTRYIQ_LOG_DB="") == "data/agent_log.duckdb"


def test_an_env_var_moves_the_gold_reader():
    assert resolved("retrieval", PANTRYIQ_GOLD_DB="/srv/gold.duckdb") == "/srv/gold.duckdb"


def test_an_env_var_moves_the_log_writer():
    """The deployed case: the log has to land on the volume, not the container's own filesystem."""
    assert resolved("log", PANTRYIQ_LOG_DB="/data/agent_log.duckdb") == "/data/agent_log.duckdb"


def test_an_empty_env_var_falls_back_rather_than_pointing_at_the_working_directory():
    """This is why both constants use `os.environ.get(name) or default` rather than
    `os.environ.get(name, default)`. The two-argument form returns `""` for a variable that is set
    but blank, and `Path("")` is `.` — so the reader would open the current *directory* as a
    database and fail with something unrecognisable. Deploy tooling sets blank variables routinely,
    and `.env.example` ships `ANTHROPIC_API_KEY=` for the same reason `claude.py` guards it.

    Both default tests above also pass a blank value, so this whole file pins the `or` form.
    """
    assert resolved("retrieval", PANTRYIQ_GOLD_DB="") == "data/pantryiq_gold.duckdb"
    assert resolved("log", PANTRYIQ_LOG_DB="") == "data/agent_log.duckdb"


def test_the_planner_follows_the_reader():
    """`planner.py` does `from retrieval import DEFAULT_DB`, so it inherits the override rather
    than needing its own. If that import ever becomes a literal path, this fails."""
    result = subprocess.run(
        [sys.executable, "-c",
         "from pantryiq.agent.planner import DEFAULT_DB; print(DEFAULT_DB)"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
             "PANTRYIQ_GOLD_DB": "/srv/gold.duckdb"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/srv/gold.duckdb"
