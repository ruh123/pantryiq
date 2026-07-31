"""Prove the quality gate BLOCKS bad data, rather than reporting it after the fact.

The brief's Phase-3 definition of done (§6) asks for a gate that stops promotion to Gold, and
says to prove it. A gate nobody has watched fail is not evidence of a gate — the same reasoning
that made §13 re-run its mutation sweep against a passing control.

Injects one deliberately impossible row into Silver, runs `dbt build`, and checks two things:

  1. the test FAILS (the error is detected at all), and
  2. downstream models are SKIPPED (the bad row does not reach Gold).

Point 2 is the one that matters. `dbt run` followed by `dbt test` would report the failure
*after* every Gold table had already been rebuilt from the bad data; `dbt build` interleaves
tests with models, so a failure stops the DAG there. That is the difference between a gate and
an alarm.

The injected row is removed in a `finally`, and the run ends by rebuilding Gold cleanly, so an
interrupted proof cannot leave poisoned data behind.

Run:  uv run python scripts/prove_gate.py
"""
from __future__ import annotations

import subprocess
import sys

import duckdb

DB = "data/pantryiq.duckdb"
BAD_ID = "__gate_probe__"
# 50,000 kcal/100 g is not "unusual", it is impossible: pure fat is ~902. A gate tuned to
# plausibility gets muted by real data (the corpus has 13 gallons of ice cream); a gate tuned to
# impossibility does not.
BAD_KCAL = 50_000.0


def dbt_build() -> tuple[int, str]:
    result = subprocess.run(
        ["uv", "run", "dbt", "build", "--profiles-dir", "."],
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def inject() -> None:
    con = duckdb.connect(DB)
    try:
        con.execute(
            """
            INSERT INTO silver.usda_foods
            (fdc_id, data_type, description_raw, canonical_name, search_text, category,
             kcal_per_100g, protein_g, fat_g, carb_g, is_deprioritized)
            VALUES (?, 'probe', 'GATE PROBE, impossible energy', 'gate probe', 'gate probe',
                    'probe', ?, 0, 0, 0, false)
            """,
            [BAD_ID, BAD_KCAL],
        )
    finally:
        con.close()


def remove() -> None:
    con = duckdb.connect(DB)
    try:
        con.execute("DELETE FROM silver.usda_foods WHERE fdc_id = ?", [BAD_ID])
    finally:
        con.close()


def main() -> int:
    print("=" * 78)
    print("PROVING THE QUALITY GATE — a deliberately impossible row must not reach Gold")
    print("=" * 78)

    code, output = dbt_build()
    print(f"\n1. clean build: exit {code}")
    if code != 0:
        print("   Gold is not clean to begin with; fix that before trusting this proof.")
        print(output[-1500:])
        return 1
    print("   PASS — the gate is green on real data, so a failure below means the probe.")

    print(f"\n2. injecting {BAD_ID} at {BAD_KCAL:,.0f} kcal/100g (pure fat is ~902)")
    inject()
    try:
        code, output = dbt_build()
    finally:
        remove()
        print("\n   probe row removed from silver.usda_foods")

    failed = code != 0
    skipped = "SKIP=0" not in output
    print(f"\n3. build with the bad row: exit {code}")
    for line in output.splitlines():
        if "accepted_range" in line and ("FAIL" in line or "ERROR" in line):
            print(f"   {line.strip()[:110]}")
        if line.strip().startswith("Done.") or "SKIP" in line and "of" in line:
            print(f"   {line.strip()[:110]}")

    print(f"\n   test failed         : {'YES' if failed else 'NO'}")
    print(f"   downstream skipped  : {'YES' if skipped else 'NO'}")

    print("\n4. rebuilding Gold from clean Silver")
    code, _ = dbt_build()
    print(f"   exit {code} — {'clean' if code == 0 else 'STILL FAILING, investigate'}")

    if failed and skipped and code == 0:
        print("\nGATE PROVEN: the bad row was detected AND blocked from reaching Gold,")
        print("and Gold rebuilds clean once it is removed.")
        return 0
    print("\nGATE NOT PROVEN — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
