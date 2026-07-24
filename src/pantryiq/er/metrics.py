"""Report the recall@k ceiling from the gold labels — the bound on everything downstream.

A scorer cannot pick an entity that candidate generation never showed it, so this number caps
2.5 and 2.7. Reported overall, per stratum, and per generator, and split by who produced the
label, because that provenance changes how much the number can be trusted (see
`docs/er_metrics.md`).

Run:  uv run python -m pantryiq.er.metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa

from pantryiq.er.candidates import recall_at_k

DEFAULT_DB = Path("data/pantryiq.duckdb")
DEFAULT_GOLD = Path("data/gold_labels")
GENERATORS = ("from_embed_search", "from_embed_desc", "from_token_head")
K_VALUES = (5, 10, 25, 50, 100)


def load_gold(gold_dir: Path | str = DEFAULT_GOLD) -> tuple[dict[str, str], dict[str, dict]]:
    """Return (normalized_text -> gold fdc_id, normalized_text -> full label record)."""
    labels: dict[str, str] = {}
    records: dict[str, dict] = {}
    for line in (Path(gold_dir) / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            labels[record["normalized_text"]] = record["fdc_id"]
            records[record["normalized_text"]] = record
    return labels, records


def load_candidates(db_path: Path | str = DEFAULT_DB) -> pa.Table:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            "SELECT normalized_text, fdc_id, rank, "
            f"{', '.join(GENERATORS)} FROM silver.ingredient_candidates"
        ).to_arrow_table()
    finally:
        con.close()


def strata(gold_dir: Path | str = DEFAULT_GOLD) -> dict[str, str]:
    return {
        row["normalized_text"]: row["frequency_stratum"]
        for row in (json.loads(line) for line in
                    (Path(gold_dir) / "sample.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip())
    }


def main() -> None:
    labels, records = load_gold()
    candidates = load_candidates()
    stratum_of = strata()
    resolvable = {text: gold for text, gold in labels.items() if gold != "no-match"}

    print(f"labels: {len(labels)}  resolvable: {len(resolvable)}  "
          f"null class: {len(labels) - len(resolvable)} "
          f"({100 * (len(labels) - len(resolvable)) / len(labels):.1f}%)")

    print("\nrecall@k (all labels)")
    for k in K_VALUES:
        print(f"  @{k:<4} {100 * recall_at_k(candidates, labels, k):5.1f}%")

    print("\nrecall@50 by stratum")
    for stratum in ("head", "mid", "tail"):
        subset = {t: g for t, g in labels.items() if stratum_of.get(t) == stratum}
        n = sum(1 for g in subset.values() if g != "no-match")
        print(f"  {stratum:5} n={n:3}  {100 * recall_at_k(candidates, subset, 50):5.1f}%")

    print("\nrecall@50 by generator (each running alone)")
    for generator in GENERATORS:
        print(f"  {generator[5:]:13} {100 * recall_at_k(candidates, labels, 50, generator):5.1f}%")

    print("\nrecall@k by label provenance")
    for labeler in ("human", "claude"):
        subset = {t: g for t, g in labels.items()
                  if records[t].get("labeler", "human") == labeler}
        n = sum(1 for g in subset.values() if g != "no-match")
        scores = "  ".join(f"@{k}={100 * recall_at_k(candidates, subset, k):.1f}%"
                           for k in (5, 25, 50))
        print(f"  {labeler:6} n={n:3}  {scores}")


if __name__ == "__main__":
    main()
