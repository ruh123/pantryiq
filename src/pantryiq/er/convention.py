"""Rule 2b (culinary default) as executable code, and an audit of the gold set against it.

`docs/labeling_guide.md` rule 1 says "prefer the least-qualified base form", which assumes a
least-qualified entry exists. Across much of USDA it does not — every milk states a fat level,
every pasta states enriched or unenriched, every green bean states raw/canned/frozen. Rule 2b
breaks that tie toward the culinary default: full-fat, enriched, raw.

This module answers "which existing labels does rule 2b overturn?" without spending human time,
so the exposure is a measured count rather than an extrapolation from the 25-string sample.

**A violation is only flagged where the labeler actually faced a fork** — where another entry
for the same food declared fewer of the attributes the line left open. Without that guard the
audit would flag `Spices, chervil, dried` for stating a form the line was silent about, but
USDA carries no raw chervil, so `dried` was never a choice and rule 2b has nothing to say.

**It flags; it does not repair.** An earlier version tried to name the replacement too, and
picked badly enough to be worth recording: `Beans, liquid from stewed kidney beans` for
`kidney bean`, `Bread, wheat` for `corn bread`, `Milk, indian buffalo, fluid` for `eagle brand
milk`. Which entry is correct is a judgment; flagged strings go back through the blind relabel
path where judgments already get made honestly.

Run:  uv run python -m pantryiq.er.convention
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels, load_sample

# The non-default value of each attribute — what rule 2b rejects when the line is silent.
# Terms are matched on word boundaries, so "light" does not fire inside "delight".
# Hyphenation is not meaningful here: USDA writes "low fat" and recipes write "low-fat", and
# treating those as different terms makes a line that states its fat level look silent.
ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "fat": ("nonfat", "low fat", "lowfat", "reduced fat", "fat free", "skim", "part skim",
            "light", "lite", "1%", "2%"),
    "fortification": ("unenriched", "unfortified"),
    "form": ("canned", "frozen", "dried", "dehydrated", "cooked", "boiled"),
}


def normalize(text: str) -> str:
    return re.sub(r"[\s-]+", " ", text.lower())


def mentions(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def violations(line: str, description: str) -> list[str]:
    """Attributes where the entity declares a non-default value the line never asked for.

    The line is treated as having spoken about an attribute if it mentions *any* of that
    attribute's terms — a line saying `skim` has chosen a fat level, so a `nonfat` entry is
    responsive to it rather than an invented qualifier.
    """
    line, description = normalize(line), normalize(description)
    found = []
    for attribute, terms in ATTRIBUTES.items():
        declared = [term for term in terms if mentions(description, term)]
        if declared and not any(mentions(line, term) for term in terms):
            found.append(attribute)
    return found


def food_key(description: str, facets: int = 2) -> tuple[str, ...]:
    """The leading facets that identify the food, before the qualifiers start.

    USDA leads with the food and then narrows: `Beans, kidney, all types, mature seeds, …`.
    One facet is too coarse to mean "the same food" — `Beans, liquid from stewed kidney beans`
    also leads with `Beans`. Two is enough to separate kidney beans from bean liquid, and
    `Spices, chervil` from the rest of the spice rack.
    """
    return tuple(part.strip().lower() for part in description.split(",")[:facets])


def candidates_by_string(con, strings: list[str]) -> dict[str, list[tuple]]:
    """(fdc_id, description, kcal, deprioritized, cosine) per labeled string."""
    rows = con.execute(
        """
        SELECT c.normalized_text, f.fdc_id, f.description_raw, f.kcal_per_100g,
               f.is_deprioritized, GREATEST(c.cosine_search, c.cosine_desc)
        FROM silver.ingredient_candidates c
        JOIN silver.usda_foods f USING (fdc_id)
        WHERE c.normalized_text IN (SELECT UNNEST(?))
        """,
        [strings],
    ).fetchall()
    grouped: dict[str, list[tuple]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row[1:])
    return grouped


def had_a_fork(line: str, current: str, candidates: list[tuple]) -> bool:
    """Whether the labeler actually faced a choice about the violated attributes.

    Some qualifiers are inherent to the food rather than chosen: USDA carries no raw chervil,
    so `Spices, chervil, dried` states a form the line was silent about but the labeler never
    picked it. Rule 2b has nothing to say there. It bites only where another entry for the
    same food declared fewer of the attributes the line left open.

    This deliberately does not name a replacement. Which entry is correct is a judgment, and
    string heuristics pick badly — an earlier version proposed `Beans, liquid from stewed
    kidney beans` for `kidney bean` and `Bread, wheat` for `corn bread`. Flagged strings go
    back through the blind relabel path instead.
    """
    key = food_key(current)
    broken = set(violations(line, current))
    return any(food_key(row[1]) == key and row[1] != current
               and set(violations(line, row[1])) < broken
               for row in candidates)


def audit(db_path: Path | str = DEFAULT_DB, gold_dir: Path | str = DEFAULT_OUT) -> list[dict]:
    """Every gold label rule 2b puts in question — flagged, not repaired."""
    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    resolved = {text: record["fdc_id"] for text, record in labels.items()
                if record["fdc_id"] != NO_MATCH}
    split_of = {row["normalized_text"]: row["split"] for row in load_sample(gold_dir)}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        description = dict(con.execute(
            "SELECT fdc_id, description_raw FROM silver.usda_foods").fetchall())
        candidates = candidates_by_string(con, list(resolved))
    finally:
        con.close()

    findings = []
    for text, fdc_id in sorted(resolved.items()):
        current = description.get(fdc_id)
        if current is None:
            continue
        broken = violations(text, current)
        if not broken or not had_a_fork(text, current, candidates.get(text, [])):
            continue
        findings.append({
            "normalized_text": text,
            "attributes": broken,
            "current_fdc_id": fdc_id,
            "current": current,
            "labeler": labels[text].get("labeler"),
            "split": split_of.get(text),
        })
    return findings


def main() -> None:
    labels = load_labels(Path(DEFAULT_OUT) / "labels.jsonl")
    resolved = sum(1 for record in labels.values() if record["fdc_id"] != NO_MATCH)
    findings = audit()

    print(f"rule 2b audit: {len(findings)} of {resolved} resolved labels flagged "
          f"({100 * len(findings) / resolved:.1f}%) — each states an attribute the line was "
          "silent about,\nwhere another entry for the same food stated fewer. Which entry is "
          "right is a judgment, not a\nstring operation: these go back through the blind "
          "relabel path.\n")
    by_attribute: dict[str, int] = {}
    by_labeler: dict[str, int] = {}
    for finding in findings:
        for attribute in finding["attributes"]:
            by_attribute[attribute] = by_attribute.get(attribute, 0) + 1
        by_labeler[finding["labeler"]] = by_labeler.get(finding["labeler"], 0) + 1
    print(f"  by attribute: {by_attribute}")
    print(f"  by labeler:   {by_labeler}\n")
    for finding in findings:
        print(f"  {finding['normalized_text']!r}  [{', '.join(finding['attributes'])}]  "
              f"{finding['split']}")
        print(f"      {finding['current']}")


if __name__ == "__main__":
    main()
