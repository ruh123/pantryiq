"""Does 2.6 earn its cost? — measured before the adjudicator is built.

The brief's 2.6 sends mid-confidence resolutions to a Claude adjudicator, on the premise that an
LLM given the labeling guide and the candidate list fixes what the embedding resolver gets wrong.
§9 of `docs/er_metrics.md` produced the first evidence against that premise: an annotator with the
full guide and 40 candidates scored 48.0% entity / 65.2% kcal-equivalent, against the shipped
resolver's 46.8% / 63.6%. Suggestive, small n, different string sets.

## ⚠️ THIS TEST DOES NOT WORK, AND THE REASON IS THE POINT

It was built on a paired-design argument: both arms scored against the same contested gold, so
shared label error cancels in the difference. That argument is **wrong here**, because the error
is not shared — it is *correlated with one arm*.

**166 of the 177 tune labels were produced by a Claude annotator following this same guide.** A
Claude adjudicator re-running that process agrees with them because it is the process that made
them; the resolver uses embeddings, an independent method, so it necessarily looks worse. Scored
that way the adjudicator "wins" by +42.4pp entity / +32.9pp kcal (p < 0.0001) — a measurement of
self-consistency, not of correctness. `docs/er_metrics.md` §1 stated this constraint before any
of this was built:

> **A Claude adjudicator (2.6) cannot be honestly scored against these labels** — that is
> self-consistency. Score it on human-labeled rows only, or do not claim it.

The only valid comparison is against the blind human relabels, and the tune split overlaps just
**11** of them (9 with kcal on both sides). At that n the honest result — adjudicator +18.2pp
entity [−18.2, +54.5], p = 0.63 — resolves nothing in either direction.

**So this module reports a non-answer, deliberately.** It is kept because the failure is worth
more than the number would have been: it demonstrates the trap concretely, and it establishes
that 2.6's value is *structurally unmeasurable* against the current gold set rather than merely
unmeasured. What it can still measure without any ground truth is **method disagreement** —
where an embedding resolver and an LLM reading the guide diverge, the string is contested, and
that flag needs no reference labels at all.

Retained design choices, valid on their own terms:

- **The tune split, not the holdout.** The holdout has been read once, in §7. Spending it on a
  build/skip decision would burn the project's only clean estimate.
- **2.6 gets its best shot** — `claude-opus-5`, the strongest annotator in §9. A null from a weak
  annotator would prove nothing.
- **The low-confidence band is reported separately**, because 2.6 fires there rather than
  everywhere.

Run:  uv run python -m pantryiq.er.adjudicator_value            # annotate the tune split
      uv run python -m pantryiq.er.adjudicator_value --report   # the paired verdict
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import duckdb
import numpy as np

from pantryiq.er.ensemble import ANNOTATORS, load_judgments, run
from pantryiq.er.ensemble import OUTPUT as JUDGMENTS
from pantryiq.er.evaluate import mcnemar, paired_bootstrap
from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels, load_sample
from pantryiq.er.nutrition import EQUIVALENT, error
from pantryiq.er.relabel import same_entity
from pantryiq.er.resolve import LOW_CONFIDENCE, load_curve, resolve

ADJUDICATOR = ANNOTATORS[0]  # claude-opus-5 — the strongest annotator in §9


def tune_strings(gold_dir: Path | str = DEFAULT_OUT) -> list[str]:
    return sorted(row["normalized_text"] for row in load_sample(gold_dir)
                  if row["split"] == "tune")


def paired_rows(gold_dir: Path | str = DEFAULT_OUT, db_path: Path | str = DEFAULT_DB):
    """One row per comparable tune string: gold, resolver pick, adjudicator pick, confidence."""
    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    judgments = load_judgments(Path(gold_dir) / JUDGMENTS)
    resolved = {text: (fdc_id, confidence)
                for text, fdc_id, _, confidence, _, _ in resolve(db_path, strings=tune_strings())}

    rows = []
    for text in tune_strings(gold_dir):
        gold = labels[text]["fdc_id"]
        judgment = judgments.get((text, ADJUDICATOR))
        if gold == NO_MATCH or judgment is None or "error" in judgment or text not in resolved:
            continue
        pick, confidence = resolved[text]
        rows.append({"text": text, "gold": gold, "resolver": pick,
                     "adjudicator": judgment["fdc_id"], "confidence": confidence})
    return rows


def compare(rows: list[dict], reference: dict[str, str], description: dict, kcal: dict,
            name: str) -> None:
    """Paired resolver-vs-adjudicator comparison against `reference`.

    The reference is an explicit argument because it is the whole ballgame: filtering rows to
    those that HAVE a human judgment does not change what they are scored against. An earlier
    version scored every section against the gold label and printed a self-consistency figure
    under the heading "the only valid comparison".
    """
    rows = [row for row in rows if row["text"] in reference]
    if not rows:
        print(f"  {name}: no comparable strings")
        return

    entity = [(same_entity(row["resolver"], reference[row["text"]], description),
               same_entity(row["adjudicator"], reference[row["text"]], description))
              for row in rows]
    kcal_pairs = []
    for row in rows:
        truth = reference[row["text"]]
        base = error(kcal.get(truth), kcal.get(row["resolver"]))
        llm = error(kcal.get(truth), kcal.get(row["adjudicator"]))
        if base is not None and llm is not None:
            kcal_pairs.append((base, llm))
    equivalence = [(base < EQUIVALENT, llm < EQUIVALENT) for base, llm in kcal_pairs]

    print(f"\n  {name}  (n={len(rows)} strings, {len(kcal_pairs)} with kcal on both sides)")
    for label, pairs in (("entity top-1", entity), ("kcal-equivalent", equivalence)):
        if not pairs:
            continue
        resolver_rate = 100 * sum(1 for base, _ in pairs if base) / len(pairs)
        llm_rate = 100 * sum(1 for _, llm in pairs if llm) / len(pairs)
        base_only, llm_only, p_value = mcnemar(pairs)
        observed, low, high, share = paired_bootstrap(
            [(float(base), float(llm)) for base, llm in pairs])
        print(f"    {label:16} resolver {resolver_rate:5.1f}%  adjudicator {llm_rate:5.1f}%   "
              f"diff {100 * observed:+5.1f}pp [{100 * low:+.1f}, {100 * high:+.1f}]")
        print(f"    {'':16} McNemar: resolver-only {base_only:3}  adjudicator-only {llm_only:3}  "
              f"exact p={p_value:.4f}   P(adjudicator better)={share:.2f}")
    if kcal_pairs:
        observed, low, high, share = paired_bootstrap(kcal_pairs, statistic=np.median)
        print(f"    {'median kcal err':16} diff {100 * observed:+5.1f}pp "
              f"[{100 * low:+.1f}, {100 * high:+.1f}]   P(adjudicator worse)={share:.2f}")


def report(gold_dir: Path | str = DEFAULT_OUT, db_path: Path | str = DEFAULT_DB) -> None:
    rows = paired_rows(gold_dir, db_path)
    if not rows:
        print("nothing to compare — run `uv run python -m pantryiq.er.adjudicator_value`")
        return
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        usda = con.execute(
            "SELECT fdc_id, description_raw, kcal_per_100g FROM silver.usda_foods").fetchall()
    finally:
        con.close()
    description = {row[0]: row[1] for row in usda}
    kcal = {row[0]: row[2] for row in usda}

    labels = load_labels(Path(gold_dir) / "labels.jsonl")
    relabels = load_labels(Path(gold_dir) / "relabel.jsonl")
    annotator_made = [row for row in rows if labels[row["text"]].get("labeler") != "human"]
    human_direct = {text for text, record in labels.items() if record.get("labeler") == "human"}
    honest = [row for row in rows
              if row["text"] in relabels and row["text"] not in human_direct]

    print("=" * 78)
    print(f"IS 2.6 WORTH BUILDING?  {ADJUDICATOR} vs the shipped resolver, tune split")
    print("=" * 78)
    print(f"gold-label provenance in this comparison set: "
          f"{len(annotator_made)}/{len(rows)} annotator-produced, "
          f"{len(rows) - len(annotator_made)} human")

    print("\n" + "!" * 78)
    print("SELF-CONSISTENCY — NOT EVIDENCE ABOUT 2.6. Reported only so the trap is visible.")
    print("!" * 78)
    print(f"{100 * len(annotator_made) / len(rows):.0f}% of these gold labels were produced by a")
    print("Claude annotator following this same guide. Scoring a Claude adjudicator against them")
    print("measures agreement between two runs of the same process. The resolver uses embeddings")
    print("— an independent method — so it necessarily scores worse. docs/er_metrics.md §1 said")
    print("this before any of it was built.")
    gold_reference = {row["text"]: row["gold"] for row in rows}
    compare(rows, gold_reference, description, kcal, "vs annotator-produced gold (INVALID)")

    print("\n" + "=" * 78)
    print("THE ONLY VALID COMPARISON — vs independent human judgment (blind relabels)")
    print("=" * 78)
    if not honest:
        print("  no strings in this set carry an independent human judgment.")
    else:
        human_reference = {text: record["fdc_id"] for text, record in relabels.items()}
        compare(honest, human_reference, description, kcal, "vs independent human judgment")
        print(f"\n  n={len(honest)}. The tune split overlaps only this many of the 25 blind")
        print("  relabels, so this comparison cannot resolve an effect of any plausible size.")
        print("  It is underpowered, not negative — do not read a null here as 'no difference'.")

    # Ground-truth-free signal: where two independent methods disagree, the string is contested.
    agree = sum(1 for row in rows
                if same_entity(row["resolver"], row["adjudicator"], description))
    flagged = [row for row in rows if row["confidence"] < LOW_CONFIDENCE]
    flagged_agree = sum(1 for row in flagged
                        if same_entity(row["resolver"], row["adjudicator"], description))
    print("\n" + "=" * 78)
    print("WHAT IS MEASURABLE WITHOUT GROUND TRUTH — method disagreement as an ambiguity flag")
    print("=" * 78)
    print(f"  resolver and adjudicator agree on {agree}/{len(rows)} "
          f"({100 * agree / len(rows):.1f}%) of tune strings")
    if flagged:
        print(f"  within the resolver's own low-confidence band: {flagged_agree}/{len(flagged)} "
              f"({100 * flagged_agree / len(flagged):.1f}%)")
    print("  These two methods are genuinely independent (embeddings vs an LLM reading the")
    print("  guide), so their disagreement flags contested strings without needing to know who")
    print("  is right — the one use of an adjudicator this label set can actually support.")


def main() -> None:
    if "--report" in sys.argv:
        report()
        return
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        raise SystemExit("ANTHROPIC_API_KEY is unset or empty — see .env")
    load_curve()  # fail early if the resolver's confidence curve has not been fitted
    asyncio.run(run(strings=tune_strings(), annotators=(ADJUDICATOR,)))
    print("\nverdict: uv run python -m pantryiq.er.adjudicator_value --report")


if __name__ == "__main__":
    main()
