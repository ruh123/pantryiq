"""Prove the quality gate BLOCKS bad data, rather than reporting it after the fact.

The brief's Phase-3 definition of done (§6) asks for a gate that stops promotion to Gold, and
says to prove it. A gate nobody has watched fail is not evidence of a gate — the same reasoning
that made §13 re-run its mutation sweep against a passing control.

Injects one deliberately impossible row into Silver and checks three things:

  1. the test FAILS (the error is detected at all),
  2. downstream models are SKIPPED (it does not propagate), and
  3. **`gold.*` is byte-identical afterwards** (it is never admitted).

Point 3 is the one that matters, and an earlier version of this script could not check it.
`dbt build` alone is only a PROPAGATION gate: dbt tests the BUILT table, so the model whose
test fails has already been materialized. Running it straight at `gold` meant the probe row
genuinely sat in `gold.canonical_ingredients`, queryable, while its own test failed — and a
real failure would leave Gold at mixed vintage indefinitely.

The build now targets `gold_staging`, and `pantryiq.gold.publish` swaps into `gold` in one
transaction only when the gate passed. So a failed gate leaves the previous Gold completely
intact, which is what "blocks promotion" has to mean.

The injected row is removed in a `finally`, and the run ends by rebuilding Gold cleanly, so an
interrupted proof cannot leave poisoned data behind.

Run:  uv run python scripts/prove_gate.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

import duckdb

from pantryiq.gold.publish import GOLD_TABLES

DB = "data/pantryiq.duckdb"
BAD_ID = "__gate_probe__"
# 50,000 kcal/100 g is not "unusual", it is impossible: pure fat is ~902. A gate tuned to
# plausibility gets muted by real data (the corpus has 13 gallons of ice cream); a gate tuned to
# impossibility does not.
BAD_KCAL = 50_000.0


def dbt_build() -> tuple[int, str]:
    result = subprocess.run(
        ["uv", "run", "dbt", "build", "--profiles-dir", ".", "--target", "staging"],
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def gold_fingerprint() -> str:
    """A hash of every published Gold table — the thing that must not move on a failed gate."""
    con = duckdb.connect(DB, read_only=True)
    try:
        parts = []
        for name in GOLD_TABLES:
            rows = con.execute(f"SELECT * FROM gold.{name}").fetchall()
            parts.append(f"{name}:{len(rows)}:{hashlib.sha256(repr(sorted(map(repr, rows))).encode()).hexdigest()}")
    finally:
        con.close()
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


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

    before = gold_fingerprint()
    code, output = dbt_build()
    print(f"\n1. clean build into staging: exit {code}")
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
    # Assert on the specific downstream models, not on the absence of a substring: a build that
    # dies before printing a summary would satisfy `"SKIP=0" not in output` while nothing was
    # actually skipped.
    skipped = all(f"SKIP relation gold_staging.{name}" in output
                  for name in ("recipe_ingredients_resolved", "recipe_nutrition", "recipe_tags"))
    after = gold_fingerprint()
    print(f"\n3. build with the bad row: exit {code}")
    for line in output.splitlines():
        if "accepted_range" in line and ("FAIL" in line or "ERROR" in line):
            print(f"   {line.strip()[:110]}")
        if line.strip().startswith("Done.") or "SKIP" in line and "of" in line:
            print(f"   {line.strip()[:110]}")

    print(f"\n   test failed          : {'YES' if failed else 'NO'}")
    print(f"   downstream skipped   : {'YES' if skipped else 'NO'}")
    print(f"   gold.* UNCHANGED     : {'YES' if before == after else 'NO'}  "
          f"({before} -> {after})")

    print("\n4. republishing from clean Silver")
    code, _ = dbt_build()
    if code == 0:
        from pantryiq.gold.publish import export, publish

        publish()
        export()
    print(f"   exit {code} — {'clean' if code == 0 else 'STILL FAILING, investigate'}")

    if failed and skipped and before == after and code == 0:
        print("\nGATE PROVEN: the bad row was detected, did not propagate, and was never")
        print("ADMITTED — gold.* is byte-identical across the failed run. Publishing resumes")
        print("once Silver is clean.")
        return 0
    print("\nGATE NOT PROVEN — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
