"""Features per (ingredient string, candidate entity) — the input to 2.5's scorer.

Deliberately eight features, not twenty. The training unit is the *pair*, so the 201 tune-split
strings expand to ~15,700 rows — but only ~165 of them are positive (one gold entity per
resolvable string), and it is the positive count that bounds how much a model can learn. Eight
features against 165 positives is already generous.

Each feature exists because something in the corpus demanded it:

- `cosine_search` / `cosine_desc` — the two embedding views measured in 2.4. Highly correlated
  (90.5% vs 88.6% recall alone) and kept both anyway: L2 regularization handles the collinearity,
  and the A/B showed they fail on different strings.
- `jaro_winkler` — character-level similarity, for the cases embeddings drift on (`soda` →
  soft drinks rather than baking soda).
- `token_jaccard` — token overlap, insensitive to word order, which is what a comma-inverted
  USDA facet string versus a recipe phrase actually differ by.
- `head_facet` — does the candidate LEAD with the string's head noun. The single strongest
  display signal from 2.3b: USDA leads with the food itself, so `Egg, …` beats `Eggnog`.
- `plain_facet_share` — share of non-leading facet words that are plain modifiers. This is the
  specificity prior the plan asked for, and it encodes guide rules 1/2b: the ordinary entry is
  the one whose extra facets are all unremarkable (`Egg, whole, raw, fresh`), while a variant
  names something (`Egg, white, dried`). Facet *count* is deliberately not used — it proved
  actively misleading, since the ordinary entry often carries MORE facets.
- `is_deprioritized` — babyfood/restaurant/brand, 14.9% of the corpus (guide rule 3).
- `cosine_margin` — this candidate's `cosine_search` minus the best available *for this string*.
  The other seven are absolute, but the decision is relative: 0.7 may be the best on offer for
  one string and mediocre for another, and a pointwise classifier cannot see that from absolute
  values. 0.0 for the leader, negative for everything else.

Run:  uv run python -m pantryiq.er.features
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa

from pantryiq.er.candidates import alias_groups
from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, PLAIN_MODIFIERS, load_labels, load_sample

FEATURES = ("cosine_search", "cosine_desc", "jaro_winkler", "token_jaccard",
            "head_facet", "plain_facet_share", "is_deprioritized", "cosine_margin")


def tokens(text: str) -> set[str]:
    return {token for token in text.lower().replace(",", " ").replace("-", " ").split() if token}


def token_jaccard(line: str, description: str) -> float:
    left, right = tokens(line), tokens(description)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def jaro_winkler(line: str, description: str) -> float:
    from rapidfuzz.distance import JaroWinkler

    return JaroWinkler.normalized_similarity(line.lower(), description.lower())


def head_facet(line: str, description: str) -> float:
    """Graded match between the string's head noun and the candidate's LEADING facet.

    1.0 when the leading facet *is* the head noun, 0.5 when it merely contains it, 0.0
    otherwise. USDA writes the food first and narrows afterwards, so leading with the head noun
    is the difference between `Egg, whole, raw` and `Eggnog` — which raw cosine gets wrong.
    """
    words = line.lower().split()
    if not words:
        return 0.0
    head = words[-1]
    leading = description.split(",")[0].strip().lower()
    if leading in (head, head + "s", head.rstrip("s")):
        return 1.0
    return 0.5 if head in leading.split() else 0.0


def plain_facet_share(description: str) -> float:
    """Share of non-leading facet words that are plain modifiers. 1.0 = fully ordinary form."""
    facets = [part.strip() for part in description.split(",")[1:] if part.strip()]
    words = [word for facet in facets for word in facet.lower().replace("-", " ").split()]
    if not words:
        return 1.0  # no qualifiers at all is maximally plain
    plain = sum(1 for word in words if word in PLAIN_MODIFIERS or word.isdigit())
    return plain / len(words)


def row_features(line: str, description: str, cosine_search: float, cosine_desc: float,
                 deprioritized: bool, cosine_margin: float = 0.0) -> dict[str, float]:
    return {
        "cosine_search": cosine_search,
        "cosine_desc": cosine_desc,
        "jaro_winkler": jaro_winkler(line, description),
        "token_jaccard": token_jaccard(line, description),
        "head_facet": head_facet(line, description),
        "plain_facet_share": plain_facet_share(description),
        "is_deprioritized": float(deprioritized),
        "cosine_margin": cosine_margin,
    }


def build(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT) -> pa.Table:
    """One row per (labeled string, candidate), with the target and the split it belongs to."""
    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    sample = load_sample(gold_dir)
    split_of = {row["normalized_text"]: row["split"] for row in sample}
    stratum_of = {row["normalized_text"]: row["frequency_stratum"] for row in sample}
    control_of = {row["normalized_text"]: bool(row.get("control", False)) for row in sample}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        aliases = alias_groups(dict(con.execute(
            "SELECT fdc_id, description_raw FROM silver.usda_foods").fetchall()))
        rows = con.execute(
            """
            SELECT c.normalized_text, f.fdc_id, f.description_raw, f.is_deprioritized,
                   c.cosine_search, c.cosine_desc, c.rank
            FROM silver.ingredient_candidates c
            JOIN silver.usda_foods f USING (fdc_id)
            WHERE c.normalized_text IN (SELECT UNNEST(?))
            ORDER BY c.normalized_text, c.rank
            """,
            [list(labels)],
        ).fetchall()
    finally:
        con.close()

    # Best cosine per string, for the relative feature. Computed over the stored candidate set,
    # which is exactly what the scorer sees at inference.
    best_cosine: dict[str, float] = {}
    for text, _, _, _, cosine_search, _, _ in rows:
        best_cosine[text] = max(best_cosine.get(text, cosine_search), cosine_search)

    columns: dict[str, list] = {key: [] for key in
                                ("normalized_text", "fdc_id", "split", "stratum", "control",
                                 "rank", "is_match", *FEATURES)}
    for text, fdc_id, description, deprioritized, cosine_search, cosine_desc, rank in rows:
        gold = labels[text]["fdc_id"]
        # A twin of the gold entity is a POSITIVE (guide rule 6). Labeling it negative would
        # teach the model that an identical description is the wrong answer.
        accept = aliases.get(gold, {gold}) if gold != NO_MATCH else set()
        columns["normalized_text"].append(text)
        columns["fdc_id"].append(fdc_id)
        columns["split"].append(split_of.get(text))
        columns["stratum"].append(stratum_of.get(text))
        columns["control"].append(control_of.get(text, False))
        columns["rank"].append(rank)
        columns["is_match"].append(fdc_id in accept)
        for key, value in row_features(text, description, cosine_search, cosine_desc,
                                       deprioritized,
                                       cosine_search - best_cosine[text]).items():
            columns[key].append(value)
    return pa.table(columns)


def main() -> None:
    table = build()
    strings = set(table.column("normalized_text").to_pylist())
    matches = table.column("is_match").to_pylist()
    splits = table.column("split").to_pylist()
    print(f"{table.num_rows:,} pairs over {len(strings)} labeled strings "
          f"({table.num_rows / len(strings):.1f} candidates each)")
    print(f"positives: {sum(matches):,} ({100 * sum(matches) / table.num_rows:.2f}% of rows)")
    for split in ("tune", "holdout"):
        rows = [index for index, value in enumerate(splits) if value == split]
        positives = sum(matches[index] for index in rows)
        in_split = {table.column("normalized_text")[index].as_py() for index in rows}
        print(f"  {split:8} {len(rows):6,} pairs  {len(in_split):3} strings  "
              f"{positives:3} positives")
    print("\nstrings with NO positive pair (gold outside the candidate set, or no-match):")
    with_positive = {table.column("normalized_text")[index].as_py()
                     for index, value in enumerate(matches) if value}
    print(f"  {len(strings - with_positive)} of {len(strings)}")


if __name__ == "__main__":
    main()
