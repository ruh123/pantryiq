"""Draw the stratified gold-label sample and freeze the train/holdout split.

The corpus is extremely skewed — 545 head strings carry 83.5% of all ingredient occurrences —
so a uniform sample would be almost entirely easy strings and would flatter the metric. We
sample a fixed number per frequency stratum and record the stratum on every row, which lets
2.5/2.7 report both unweighted and occurrence-weighted precision/recall.

The train/holdout split is assigned HERE, at sampling time, before any label exists — so it
cannot be influenced by how the labels turn out. The holdout fingerprint is written to
`manifest.json`; if it ever changes, previously reported metrics are void.

See `docs/labeling_guide.md` for what a correct label is.

Run directly:  uv run python -m pantryiq.er.gold
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import duckdb

DEFAULT_DB = Path("data/pantryiq.duckdb")
DEFAULT_OUT = Path("data/gold_labels")
SEED = 20260724

# 500 labels total. Each stratum holds far more distinct strings than we draw
# (head 545 / mid 1,580 / tail 7,228), so every draw is a genuine subsample.
SAMPLE_PER_STRATUM = {"head": 167, "mid": 167, "tail": 166}
TUNE_PER_STRATUM = {"head": 100, "mid": 100, "tail": 100}  # remainder -> frozen holdout
EXAMPLES_PER_STRING = 3

# --- Reduction to 300 (2026-07-24) ---------------------------------------------------------
# 500 proved to cost 8-12 hours of human judgment, not the ~90 minutes originally estimated.
# The reduced set is taken as a PREFIX of the existing per-stratum draw rather than re-drawn:
# the 500 was already shuffled within each stratum, so a prefix is still a random subsample,
# and it preserves labels already collected. Re-drawing with a smaller size would not.
REDUCED_PER_STRATUM = {"head": 100, "mid": 100, "tail": 100}
# Splits are interleaved (every 3rd row) rather than assigned by position, so that stopping
# early still yields both tune and holdout in proportion instead of all tune and no holdout.
HOLDOUT_EVERY = 3
# Share of rows presented with candidates SCRAMBLED and unranked, to measure how much the
# ranked display anchors the labeler. Without a control, an accept rate cannot be told apart
# from agreement.
CONTROL_SHARE = 15


def is_control(normalized_text: str, share: int = CONTROL_SHARE) -> bool:
    """Deterministic per-string control assignment — stable across re-runs."""
    digest = hashlib.sha256(f"control:{normalized_text}".encode()).hexdigest()
    return int(digest[:8], 16) % 100 < share


def reduce_sample(rows: list[dict], per_stratum: dict[str, int] | None = None) -> list[dict]:
    """Cut the sample to `per_stratum` per stratum, re-splitting and tagging controls."""
    per_stratum = per_stratum or REDUCED_PER_STRATUM
    reduced: list[dict] = []
    for stratum, size in per_stratum.items():
        in_stratum = [row for row in rows if row["frequency_stratum"] == stratum][:size]
        if len(in_stratum) < size:
            raise ValueError(f"stratum {stratum!r} has {len(in_stratum)} rows, need {size}")
        for index, row in enumerate(in_stratum):
            row = dict(row)
            row["split"] = "holdout" if index % HOLDOUT_EVERY == 2 else "tune"
            row["control"] = is_control(row["normalized_text"])
            reduced.append(row)
    return reduced


def draw_sample(db_path: Path | str = DEFAULT_DB, seed: int = SEED) -> list[dict]:
    """Return the seeded stratified sample, each row tagged with its split."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows: list[dict] = []
        for stratum, size in SAMPLE_PER_STRATUM.items():
            population = con.execute(
                "SELECT normalized_text, occurrence_count FROM silver.distinct_ingredient_strings "
                "WHERE frequency_stratum = ? ORDER BY normalized_text",
                [stratum],
            ).fetchall()
            if len(population) < size:
                raise ValueError(
                    f"stratum {stratum!r} has {len(population)} strings, need {size}"
                )
            drawn = random.Random(f"{seed}:{stratum}").sample(population, size)
            # Shuffle before splitting so the split is independent of the draw order.
            random.Random(f"{seed}:{stratum}:split").shuffle(drawn)
            for index, (text, count) in enumerate(drawn):
                rows.append(
                    {
                        "normalized_text": text,
                        "occurrence_count": count,
                        "frequency_stratum": stratum,
                        "split": "tune" if index < TUNE_PER_STRATUM[stratum] else "holdout",
                        "example_lines": _examples(con, text),
                    }
                )
        return rows
    finally:
        con.close()


def _examples(con: duckdb.DuckDBPyConnection, normalized_text: str) -> list[str]:
    """A few raw lines behind the string — 'sugar' alone is ambiguous without context."""
    return [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT line_raw FROM silver.recipe_ingredient_lines "
            "WHERE normalized_text = ? ORDER BY line_raw LIMIT ?",
            [normalized_text, EXAMPLES_PER_STRING],
        ).fetchall()
    ]


def holdout_fingerprint(rows: list[dict]) -> str:
    """Stable hash of the holdout membership — proof the frozen split never moved."""
    texts = sorted(r["normalized_text"] for r in rows if r["split"] == "holdout")
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def write_sample(rows: list[dict], out_dir: Path | str = DEFAULT_OUT) -> Path:
    """Write sample.jsonl (one row per string) plus manifest.json (the frozen fingerprint)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_stratum: dict[str, int] = {}
    for row in rows:
        per_stratum[row["frequency_stratum"]] = per_stratum.get(row["frequency_stratum"], 0) + 1
    manifest = {
        "seed": SEED,
        "total": len(rows),
        "per_stratum": per_stratum,
        "tune": sum(r["split"] == "tune" for r in rows),
        "holdout": sum(r["split"] == "holdout" for r in rows),
        "control": sum(r.get("control", False) for r in rows),
        "holdout_fingerprint": holdout_fingerprint(rows),
        "labeling_guide": "docs/labeling_guide.md",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return sample_path


def reduce_existing(out_dir: Path | str = DEFAULT_OUT) -> list[dict]:
    """Rewrite an existing sample.jsonl as the reduced set, preserving row identity."""
    out_dir = Path(out_dir)
    existing = [
        json.loads(line)
        for line in (out_dir / "sample.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    reduced = reduce_sample(existing)
    write_sample(reduced, out_dir)
    return reduced


def main() -> None:
    rows = draw_sample()
    path = write_sample(rows)
    print(f"{path}: {len(rows)} strings to label")
    for stratum in SAMPLE_PER_STRATUM:
        in_stratum = [r for r in rows if r["frequency_stratum"] == stratum]
        tune = sum(r["split"] == "tune" for r in in_stratum)
        print(f"  {stratum:5} {len(in_stratum):3}  (tune {tune}, holdout {len(in_stratum) - tune})")
    print(f"  tune {sum(r['split'] == 'tune' for r in rows)} / "
          f"holdout {sum(r['split'] == 'holdout' for r in rows)}")
    print(f"  holdout fingerprint: {holdout_fingerprint(rows)[:16]}…")


if __name__ == "__main__":
    main()
