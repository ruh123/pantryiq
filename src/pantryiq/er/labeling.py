"""Interactive labeling CLI for the gold set — see `docs/labeling_guide.md` for the rules.

Shows each sampled string with the raw lines it came from and its top candidates, and records
one judgment per string. Three things here exist to keep the resulting metric honest:

- **Search is always available.** If the labeler could only pick from the displayed list,
  recall@k would be 100% by construction and the 2.4 ceiling would be meaningless. `s`
  searches all 8,187 foods, and every label records whether it came from the list or a search.
- **The split is never displayed.** Knowing a row is holdout could bias how carefully it is
  judged.
- **The sample is verified before labeling starts.** Labels key on `normalized_text`, so if
  the parser changed since sampling, the labels would silently orphan. Both the manifest
  fingerprint and the strings' continued existence in Silver are checked.

Progress is appended after every judgment, so quitting and resuming is safe.

Run:  uv run python -m pantryiq.er.labeling
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pantryiq.er.gold import DEFAULT_OUT, holdout_fingerprint

DEFAULT_DB = Path("data/pantryiq.duckdb")
DISPLAY_K = 25
SEARCH_RESULTS = 20
NO_MATCH = "no-match"


def load_sample(out_dir: Path | str = DEFAULT_OUT) -> list[dict]:
    path = Path(out_dir) / "sample.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_sample(sample: list[dict], con, out_dir: Path | str = DEFAULT_OUT) -> None:
    """Refuse to label if the sample drifted from its manifest or from Silver.

    Labels join back on `normalized_text`. A parser change between sampling and labeling would
    orphan every label — hours of human judgment silently detached from the corpus.
    """
    manifest = json.loads((Path(out_dir) / "manifest.json").read_text())
    actual = holdout_fingerprint(sample)
    if actual != manifest["holdout_fingerprint"]:
        raise SystemExit(
            f"sample.jsonl does not match manifest.json\n"
            f"  manifest: {manifest['holdout_fingerprint']}\n  sample:   {actual}\n"
            "The frozen holdout moved — re-draw the sample or restore the file."
        )

    known = {
        row[0] for row in con.execute(
            "SELECT normalized_text FROM silver.distinct_ingredient_strings"
        ).fetchall()
    }
    missing = [row["normalized_text"] for row in sample if row["normalized_text"] not in known]
    if missing:
        raise SystemExit(
            f"{len(missing)} sampled strings no longer exist in silver."
            f"distinct_ingredient_strings (e.g. {missing[:3]}).\n"
            "Normalization changed after sampling; labels would not join. Re-run 2.1 + 2.3a."
        )


def load_labels(path: Path | str) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            labels[record["normalized_text"]] = record  # a later judgment supersedes an earlier
    return labels


def append_label(path: Path | str, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_label(normalized_text: str, fdc_id: str, via: str, rank: int | None = None) -> dict:
    return {
        "normalized_text": normalized_text,
        "fdc_id": fdc_id,
        "via": via,
        "candidate_rank": rank,
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def candidates_for(con, normalized_text: str, limit: int = DISPLAY_K) -> list[tuple]:
    return con.execute(
        """
        SELECT f.fdc_id, f.description_raw, f.kcal_per_100g, f.is_deprioritized, c.rank
        FROM silver.ingredient_candidates c
        JOIN silver.usda_foods f USING (fdc_id)
        WHERE c.normalized_text = ?
        ORDER BY c.rank
        LIMIT ?
        """,
        [normalized_text, limit],
    ).fetchall()


def search_foods(con, query: str, limit: int = SEARCH_RESULTS) -> list[tuple]:
    """Substring search over every USDA food, shortest (most generic) description first."""
    tokens = [token for token in query.lower().split() if token]
    if not tokens:
        return []
    where = " AND ".join(["lower(description_raw) LIKE ?"] * len(tokens))
    return con.execute(
        f"SELECT fdc_id, description_raw, kcal_per_100g, is_deprioritized, NULL "  # noqa: S608
        f"FROM silver.usda_foods WHERE {where} "
        "ORDER BY length(description_raw), description_raw LIMIT ?",
        [f"%{token}%" for token in tokens] + [limit],
    ).fetchall()


def _show(rows: list[tuple]) -> None:
    for position, (_, description, kcal, deprioritized, _rank) in enumerate(rows):
        energy = f"{kcal:>4.0f} kcal" if kcal is not None else "  no kcal"
        flag = " ~" if deprioritized else "  "
        print(f"  {position:>3}{flag}{energy}  {description[:88]}")


def _prompt_search(con) -> tuple[str, int | None] | None:
    query = input("  search> ").strip()
    if not query:
        return None
    results = search_foods(con, query)
    if not results:
        print("  no matches")
        return None
    _show(results)
    choice = input("  pick number (blank to cancel)> ").strip()
    if choice.isdigit() and int(choice) < len(results):
        return results[int(choice)][0], None
    return None


def main() -> None:
    con = duckdb.connect(str(DEFAULT_DB), read_only=True)
    labels_path = Path(DEFAULT_OUT) / "labels.jsonl"
    try:
        sample = load_sample()
        verify_sample(sample, con)
        labels = load_labels(labels_path)
        todo = [row for row in sample if row["normalized_text"] not in labels]

        print(f"{len(labels)} of {len(sample)} labeled — {len(todo)} to go")
        print("keys: number = pick   n = no-match   s = search all foods   k = skip   q = quit")
        print("rules: docs/labeling_guide.md   (~ = deprioritized: babyfood/restaurant/brand)\n")

        for position, row in enumerate(todo, start=1):
            text = row["normalized_text"]
            print(f"[{position}/{len(todo)}] {row['frequency_stratum']} · "
                  f"{row['occurrence_count']} occurrences")
            print(f"  STRING: {text!r}")
            for line in row["example_lines"]:
                print(f"    seen as: {line}")
            rows = candidates_for(con, text)
            _show(rows)

            while True:
                choice = input("  > ").strip().lower()
                if choice == "q":
                    print(f"\nsaved {len(load_labels(labels_path))} labels to {labels_path}")
                    return
                if choice == "k":
                    break
                if choice == "n":
                    append_label(labels_path, make_label(text, NO_MATCH, "candidate"))
                    break
                if choice == "s":
                    picked = _prompt_search(con)
                    if picked:
                        append_label(labels_path, make_label(text, picked[0], "search"))
                        break
                    continue
                if choice.isdigit() and int(choice) < len(rows):
                    chosen = rows[int(choice)]
                    append_label(labels_path, make_label(text, chosen[0], "candidate", chosen[4]))
                    break
                print("  ?")
            print()

        print(f"done — {len(load_labels(labels_path))} labels in {labels_path}")
    except (KeyboardInterrupt, EOFError):
        print(f"\ninterrupted — {len(load_labels(labels_path))} labels saved")
    finally:
        con.close()


if __name__ == "__main__":
    main()
