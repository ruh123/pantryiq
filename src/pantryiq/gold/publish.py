"""Publish a gated Gold build, and stamp it — the missing half of write-audit-publish.

`dbt build` is a *propagation* gate: a failing test skips everything downstream, but the model
whose test failed has already been materialized, because dbt tests the BUILT table. So the old
flow left Gold at **mixed vintage** on a partial failure — `canonical_ingredients` new (and
possibly poisoned), `recipe_nutrition` from the previous run — with every join key stable across
vintages, every bound satisfied, every uniqueness and referential test passing. Nothing in the
data said which run produced which table, so a half-built warehouse was indistinguishable from
a good one. That is this project's own failure mode wearing an orchestration hat.

The fix is a real publish step:

    dbt build --target staging     -> gold_staging.*, tested there
    publish                        -> swap into gold.* in ONE transaction, + a run stamp

Gold therefore only ever changes all-at-once, from a build whose tests all passed. A failed gate
leaves the *previous* Gold completely intact rather than partly replaced.

`gold.pipeline_run` is the vintage record: one row per publish, with the source row counts the
build was derived from. Every number a consumer reads now has an `as_of` next to its
`data_trust_score` — a quality signal without a freshness signal was only ever half a signal.

Run:  uv run python -m pantryiq.gold.publish
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_DB = Path("data/pantryiq.duckdb")
STAGING_SCHEMA = "gold_staging"
PUBLISHED_SCHEMA = "gold"
EXPORT_PATH = Path("data/pantryiq_gold.duckdb")

# The published surface. Anything not listed here is build scratch and is not promoted.
GOLD_TABLES = (
    "canonical_ingredients",
    "ingredient_costs",
    "recipe_ingredients_resolved",
    "recipe_nutrition",
    "recipe_tags",
    "recipe_meta",
)
# Counted into the stamp so a consumer can tell which Silver vintage Gold was derived from.
SOURCE_TABLES = (
    "silver.recipe_ingredient_lines",
    "silver.ingredient_entity_map",
    "silver.recipe_ingredient_grams",
    "silver.usda_foods",
    "silver.usda_portions",
    "silver.recipe_servings",
    "silver.recipe_meta",
    # The two the dietary tags rest on. Both were missing from the vintage record while being
    # load-bearing for the project's highest-consequence claim.
    "silver.usda_food_categories",
    "silver.entity_dietary_flags",
)


def build_staging(db_path: Path | str = DEFAULT_DB) -> None:
    """Run the gated build into the staging schema. Raises if the gate fails."""
    result = subprocess.run(
        ["uv", "run", "dbt", "build", "--profiles-dir", ".", "--target", "staging"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3],
    )
    if result.returncode != 0:
        raise RuntimeError(
            "quality gate FAILED — nothing published, previous Gold left intact.\n"
            + result.stdout[-2000:]
        )


def _git_sha() -> str:
    """The commit this build came from, marked `-dirty` when the tree does not match it.

    Without the suffix the stamp names a commit that cannot have produced the data: the published
    tags were allowlist output while `gold.pipeline_run.git_sha` pointed at a commit whose
    `recipe_tags.sql` still held the old denylist. A provenance record that can be wrong about
    provenance is worse than none, because it is trusted.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def publish(db_path: Path | str = DEFAULT_DB) -> dict:
    """Swap staging into `gold` in one transaction and stamp the run.

    Every table moves together or none does. A partial publish is the exact failure this
    function exists to prevent, so the swap is never done table-by-table outside a transaction.
    """
    con = duckdb.connect(str(db_path))
    try:
        missing = [name for name in GOLD_TABLES if not con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [STAGING_SCHEMA, name]).fetchone()[0]]
        if missing:
            raise RuntimeError(f"staging is incomplete, refusing to publish: {missing}")

        counts = {name: con.execute(
            f"SELECT count(*) FROM {STAGING_SCHEMA}.{name}").fetchone()[0] for name in GOLD_TABLES}
        sources = {name: con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                   for name in SOURCE_TABLES}

        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {PUBLISHED_SCHEMA}")
            for name in GOLD_TABLES:
                con.execute(f"CREATE OR REPLACE TABLE {PUBLISHED_SCHEMA}.{name} AS "
                            f"SELECT * FROM {STAGING_SCHEMA}.{name}")
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {PUBLISHED_SCHEMA}.pipeline_run AS
                SELECT ? AS published_at, ? AS git_sha, ? AS source_row_counts,
                       ? AS gold_row_counts
                """,
                [datetime.now(timezone.utc), _git_sha(), str(sources), str(counts)],
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    finally:
        con.close()
    return counts


def export(db_path: Path | str = DEFAULT_DB, out_path: Path | str = EXPORT_PATH) -> Path:
    """Write Gold alone to its own file — the deploy artifact §4 asks for.

    Silver is ~870K rows dominated by `ingredient_candidates` (715K) which is pure build
    scratch; shipping the shared file would ship all of it. Serving from a separate file also
    takes the reader out of the writer's lock: a Streamlit session on the shared file blocks the
    pipeline's next write, which is the wrong direction for a failure to travel.
    """
    out_path = Path(out_path).resolve()
    out_path.unlink(missing_ok=True)
    if "'" in str(out_path):
        # ATTACH takes no bound parameters, so the path is interpolated. A quote in it would
        # break out of the literal; refuse rather than build a broken statement.
        raise ValueError(f"export path may not contain a single quote: {out_path}")
    source = Path(db_path).resolve()
    if "'" in str(source):
        raise ValueError(f"database path may not contain a single quote: {source}")
    # The NEW file is the primary connection and the warehouse is attached READ_ONLY, rather
    # than the other way round: a read-only primary cannot create the export, and a writable
    # one would take the warehouse's exclusive lock just to copy out of it.
    con = duckdb.connect(str(out_path))
    try:
        con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")  # noqa: S608 — quote-checked
        con.execute("CREATE SCHEMA IF NOT EXISTS gold")
        for name in (*GOLD_TABLES, "pipeline_run"):
            con.execute(f"CREATE TABLE gold.{name} AS "
                        f"SELECT * FROM src.{PUBLISHED_SCHEMA}.{name}")
        con.execute("DETACH src")
    finally:
        con.close()
    return out_path


def main() -> None:
    from pantryiq.lakehouse.writelock import warehouse_lock

    print("building into staging (gated) ...")
    # Held across the whole gate-and-swap, not just the swap: dbt is a writer too, and a
    # concurrent Silver step between the build and the publish would produce exactly the
    # mixed-vintage Gold this module exists to prevent.
    with warehouse_lock():
        build_staging()
        counts = publish()
        path = export()

    print(f"PUBLISHED — all {len(GOLD_TABLES)} tables swapped in one transaction")
    for name, count in counts.items():
        print(f"  gold.{name}: {count:,}")
    print(f"  exported: {path} ({path.stat().st_size / 1e6:.1f} MB, Gold only)")
    print("  gold.pipeline_run records the vintage: published_at, git sha, source row counts.")


if __name__ == "__main__":
    main()
