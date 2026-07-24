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
import random
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pantryiq.er.gold import DEFAULT_OUT, holdout_fingerprint

DEFAULT_DB = Path("data/pantryiq.duckdb")
DISPLAY_K = 25
SHORTLIST = 5  # shown by default; `m` expands to DISPLAY_K
SEARCH_RESULTS = 20
NO_MATCH = "no-match"

# Facet words that describe the ordinary form of a food rather than naming a variant.
# "Egg, whole, raw, fresh" is all-plain; "Egg, white, dried" is not.
PLAIN_MODIFIERS = {
    "raw", "whole", "fresh", "fluid", "regular", "plain", "unprepared", "all-purpose",
    "enriched", "bleached", "unbleached", "table", "salted", "granulated", "cultured",
    "full", "fat", "with", "without", "and", "commercial", "stick", "light", "large",
    "or", "added", "solids", "unsalted",
}


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


def make_label(normalized_text: str, fdc_id: str, via: str, rank: int | None = None,
               display_rank: int | None = None, presentation: str = "ranked",
               suggested_fdc_id: str | None = None) -> dict:
    """One judgment.

    `candidate_rank` is the generation rank (what recall@k is measured over); `display_rank`
    is where it sat on screen. `presentation` records whether the candidates were ranked or
    scrambled, and `suggested_fdc_id` what the model would have proposed — together these let
    us report agreement separately for ranked and unranked presentation, which is the only way
    to tell genuine agreement from reflex acceptance of the top row.
    """
    return {
        "normalized_text": normalized_text,
        "fdc_id": fdc_id,
        "via": via,
        "candidate_rank": rank,
        "display_rank": display_rank,
        "presentation": presentation,
        "suggested_fdc_id": suggested_fdc_id,
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def display_score(query: str, cosine: float, description: str, deprioritized: bool,
                  has_kcal: bool) -> float:
    """Order candidates for a human, encoding the guide's base-form convention.

    Presentation only — the recall@k ceiling is measured over the stored generation ranks, so
    this cannot move any metric. It exists because raw cosine buries the right answer: it put
    `Eggnog` above `Egg, whole, raw, fresh` and `DENNY'S, onion rings` above `Onions, raw`.

    The load-bearing signals are the FIRST facet matching the head noun (USDA leads with the
    food itself, so "Egg, ..." beats "Eggnog") and every remaining facet being a plain
    modifier. Facet *count* is deliberately not penalized — it proved actively misleading,
    since the ordinary entry often carries more qualifiers ("Egg, whole, raw, fresh") than a
    rare variant ("Egg, white, dried").
    """
    words = query.split()
    head = words[-1] if words else ""
    facets = [part.strip().lower() for part in description.split(",") if part.strip()]
    first = facets[0] if facets else ""
    tokens = set(" ".join(facets).replace("-", " ").split())

    score = cosine
    leads = first in (head, head + "s")
    if leads:
        score += 0.30
    elif head in first.split():
        score += 0.18
    if head in tokens:
        score += 0.08
    if words and all(word in tokens for word in words):
        score += 0.10

    # Bonuses apply only to candidates that are lexically relevant at all. Ungated, the
    # plain-modifier bonus reorders the list by "plainness" when NOTHING matches — the string
    # "dream whip" was headed by Arrowhead/Taro/Pummelo purely because they are "..., raw".
    # For a hopeless string the honest display is plain cosine order.
    if leads or head in tokens:
        qualifiers = [t for facet in facets[1:] for t in facet.replace("-", " ").split()]
        if qualifiers and all(t in PLAIN_MODIFIERS or t.isdigit() for t in qualifiers):
            score += 0.28
        if has_kcal:
            score += 0.03
    if deprioritized:
        score -= 0.15
    return score


def candidates_for(con, normalized_text: str, limit: int = DISPLAY_K) -> list[tuple]:
    """Top candidates for display: deduplicated by description, base-form-first."""
    rows = con.execute(
        """
        SELECT f.fdc_id, f.description_raw, f.kcal_per_100g, f.is_deprioritized, c.rank,
               GREATEST(c.cosine_search, c.cosine_desc)
        FROM silver.ingredient_candidates c
        JOIN silver.usda_foods f USING (fdc_id)
        WHERE c.normalized_text = ?
        ORDER BY c.rank
        """,
        [normalized_text],
    ).fetchall()

    # 94 descriptions exist twice (Foundation + SR Legacy); showing both wastes a slot.
    seen: set[str] = set()
    unique = [row for row in rows if not (row[1] in seen or seen.add(row[1]))]
    unique.sort(
        key=lambda row: -display_score(normalized_text, row[5], row[1], row[3], row[2] is not None)
    )
    return unique[:limit]


def search_foods(con, query: str, limit: int = SEARCH_RESULTS) -> list[tuple]:
    """Search every USDA food: exact substring matches first, then fuzzy.

    Fuzzy matters because the labeler has to guess USDA's vocabulary — their word for sugar is
    `Sugars, granulated`, and a strict substring search punishes every near miss.
    """
    tokens = [token for token in query.lower().split() if token]
    if not tokens:
        return []
    where = " AND ".join(["lower(description_raw) LIKE ?"] * len(tokens))
    exact = con.execute(
        "SELECT fdc_id, description_raw, kcal_per_100g, is_deprioritized, NULL "  # noqa: S608
        f"FROM silver.usda_foods WHERE {where} "
        "ORDER BY length(description_raw), description_raw LIMIT ?",
        [f"%{token}%" for token in tokens] + [limit],
    ).fetchall()
    if len(exact) >= limit:
        return exact

    from rapidfuzz import fuzz, process

    everything = con.execute(
        "SELECT fdc_id, description_raw, kcal_per_100g, is_deprioritized FROM silver.usda_foods"
    ).fetchall()
    found = {row[0] for row in exact}
    # token_set_ratio, compared against the alternatives on real queries: WRatio rewards long
    # descriptions ("chedder cheese" -> pasteurized process cheese food), partial_ratio matches
    # substrings anywhere ("brown suger" -> instant oatmeal). token_set handles word-order and
    # subset matches, which is what a half-remembered USDA name actually looks like.
    ranked = process.extract(
        query, {i: row[1] for i, row in enumerate(everything)},
        scorer=fuzz.token_set_ratio, limit=limit * 3,
    )
    for _, _, index in ranked:
        row = everything[index]
        if row[0] not in found:
            found.add(row[0])
            exact.append((*row, None))
        if len(exact) >= limit:
            break
    return exact


WEAK_MATCH_COSINE = 0.55


def _show(rows: list[tuple]) -> None:
    for position, row in enumerate(rows):
        _, description, kcal, deprioritized = row[0], row[1], row[2], row[3]
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
        print("keys: ENTER = accept the top row   number = pick   n = no-match")
        print("      s = search all foods   m = show more   k = skip   q = quit")
        print("rules: docs/labeling_guide.md   (~ = deprioritized: babyfood/restaurant/brand)\n")

        for position, row in enumerate(todo, start=1):
            text = row["normalized_text"]
            control = row.get("control", False)
            ranked = candidates_for(con, text)
            suggested = ranked[0][0] if ranked else None
            presentation = "scrambled" if control else "ranked"

            print(f"[{position}/{len(todo)}] {row['frequency_stratum']} · "
                  f"{row['occurrence_count']} occurrences")
            print(f"  STRING: {text!r}")
            for line in row["example_lines"]:
                print(f"    seen as: {line}")

            # A hint, not a verdict — the judgment stays the labeler's.
            if ranked and max(row[5] for row in ranked) < WEAK_MATCH_COSINE:
                print("  (nothing similar in USDA — 'n' is likely right)")

            shown = list(ranked[:SHORTLIST])
            if control:
                # No suggestion, no ranking signal — this row measures anchoring.
                random.Random(f"scramble:{text}").shuffle(shown)
                print("  (unranked — no suggestion on this one)")
            _show(shown)
            prompt = ("  [n]o-match [s]earch [m]ore [k]skip [q]uit > " if control else
                      "  ENTER accepts 0   [n]o-match [s]earch [m]ore [k]skip [q]uit > ")

            while True:
                choice = input(prompt).strip().lower()
                if choice == "q":
                    print(f"\nsaved {len(load_labels(labels_path))} labels to {labels_path}")
                    return
                if choice == "k":
                    break
                if choice == "" and not control and shown:
                    chosen = shown[0]
                    append_label(labels_path, make_label(
                        text, chosen[0], "candidate", chosen[4], 0, presentation, suggested))
                    break
                if choice == "m":
                    shown = list(ranked)
                    if control:
                        random.Random(f"scramble:{text}").shuffle(shown)
                    _show(shown)
                    continue
                if choice == "n":
                    append_label(labels_path, make_label(
                        text, NO_MATCH, "candidate", None, None, presentation, suggested))
                    break
                if choice == "s":
                    picked = _prompt_search(con)
                    if picked:
                        append_label(labels_path, make_label(
                            text, picked[0], "search", None, None, presentation, suggested))
                        break
                    continue
                if choice.isdigit() and int(choice) < len(shown):
                    chosen = shown[int(choice)]
                    append_label(labels_path, make_label(
                        text, chosen[0], "candidate", chosen[4], int(choice),
                        presentation, suggested))
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
