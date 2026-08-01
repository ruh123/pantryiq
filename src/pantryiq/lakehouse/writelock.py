"""One advisory lock around every writer of the DuckDB warehouse.

DuckDB takes an exclusive OS lock on the file, so two writers do not interleave — the second
one fails. Airflow's `max_active_tasks=1` bounds concurrency *within* a DAG and `max_active_runs`
bounds runs of the *same* DAG; neither is cross-DAG, and neither exists at all for a human
running `make gold` or `python -m pantryiq.silver.grams` in a terminal while the pipeline runs.

An `flock` covers all of those, because it lives with the process rather than the scheduler.
It is advisory — it only binds code that asks for it — so every `main()` that writes takes it.

The failure it prevents is not corruption (DuckDB refuses cleanly); it is a confusing mid-run
`IOException` that Airflow then retries straight back into the same contention.

Serving is deliberately NOT in this contention set: Phase 4 reads the exported
`data/pantryiq_gold.duckdb`, which no writer ever touches.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path("data/.warehouse.lock")


@contextmanager
def warehouse_lock(path: Path | str = LOCK_PATH, blocking: bool = True):
    """Hold the warehouse write lock for the duration of the block.

    Blocking by default: a queued writer is almost always better than a failed one, since these
    are batch steps. Pass `blocking=False` to fail fast instead, which is what a human at a
    terminal usually wants.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another process is writing the warehouse (pipeline run, dbt build, or a "
                "manual module). Wait for it rather than running both."
            ) from error
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
