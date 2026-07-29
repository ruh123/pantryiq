"""Blind A/B adjudication of the relabel disagreements.

The blind relabel pass produced 36% agreement, but the disagreements are dominated by *facet*
choices (`evaporated milk` → regular or nonfat?) rather than by picking the wrong food. That
leaves three very different explanations tangled together, and they imply different things for
2.5's precision metric:

- **annotator error** — the gold labels are unreliable and 2.5 is measuring noise
- **convention drift** — both passes are defensible, they applied `docs/labeling_guide.md`
  differently, and the metric is measuring conformity to a convention
- **genuine ambiguity** — the ingredient string does not determine one USDA entity at all, so
  no single-entity precision number can be honest

This pass separates them. Each disagreement is shown as two descriptions, **A/B in randomized
order with no indication of which pass produced which**, and the question asked is not "which
do you prefer" but "which one does the guide select". Blindness matters more here than in the
relabel pass: without it this becomes the labeler defending their own earlier judgment.

The A/B order is seeded per string, so the presentation is reproducible and every record keeps
`shown_first` — the mapping can be re-derived and audited afterwards.

Neither `labels.jsonl` nor `relabel.jsonl` is modified. This measures; it does not repair.

Run:  uv run python -m pantryiq.er.adjudicate            # adjudicate
      uv run python -m pantryiq.er.adjudicate --report   # the three-way split
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import DEFAULT_DB, NO_MATCH, load_labels, load_sample
from pantryiq.er.relabel import ANNOTATOR, agreement, descriptions

HUMAN = "human"
AMBIGUOUS = "ambiguous"
NO_MATCH_DISPLAY = "— nothing in USDA is a reasonable nutritional stand-in —"


def order_for(text: str) -> bool:
    """True when the annotator's label is shown as A. Seeded per string, so it reproduces."""
    return random.Random(f"adjudicate:{text}").random() < 0.5


def option_text(fdc_id: str, description: dict[str, str],
                kcal: dict[str, float] | None = None) -> str:
    """One option as displayed. Energy is shown only when `kcal` is supplied — it is real
    decision-relevant information while judging, and noise in the summary."""
    if fdc_id == NO_MATCH:
        return NO_MATCH_DISPLAY
    if kcal is None:
        return f"{description.get(fdc_id, '?')} ({fdc_id})"
    energy = kcal.get(fdc_id)
    suffix = f"  [{energy:.0f} kcal/100g]" if energy is not None else "  [no kcal]"
    return f"{description.get(fdc_id, '?')}{suffix}"


def make_verdict(text: str, claude_fdc: str, human_fdc: str, choice: str) -> dict:
    """One adjudication. `choice` is 'a', 'b', or AMBIGUOUS."""
    claude_first = order_for(text)
    if choice == AMBIGUOUS:
        verdict = AMBIGUOUS
    else:
        chose_a = choice == "a"
        verdict = ANNOTATOR if chose_a == claude_first else HUMAN
    return {
        "normalized_text": text,
        "verdict": verdict,
        "claude_fdc_id": claude_fdc,
        "human_fdc_id": human_fdc,
        "shown_first": ANNOTATOR if claude_first else HUMAN,
        "chose_position": None if choice == AMBIGUOUS else choice,
        "adjudicated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_verdicts(path: Path | str) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    return {
        record["normalized_text"]: record
        for record in (json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def pending(out_dir: Path | str, description: dict[str, str]) -> list[tuple[str, str, str]]:
    """Disagreements not yet adjudicated, in a stable order."""
    original = load_labels(Path(out_dir) / "labels.jsonl")
    relabeled = load_labels(Path(out_dir) / "relabel.jsonl")
    done = load_verdicts(Path(out_dir) / "adjudication.jsonl")
    _, _, conflicts = agreement(original, relabeled, description)
    return [row for row in conflicts if row[0] not in done]


def report(out_dir: Path | str = DEFAULT_OUT, db_path: Path | str = DEFAULT_DB) -> None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        description = descriptions(con)
    finally:
        con.close()

    original = load_labels(Path(out_dir) / "labels.jsonl")
    relabeled = load_labels(Path(out_dir) / "relabel.jsonl")
    verdicts = load_verdicts(Path(out_dir) / "adjudication.jsonl")
    agreed, compared, conflicts = agreement(original, relabeled, description)
    if not verdicts:
        print("nothing adjudicated yet — run `uv run python -m pantryiq.er.adjudicate`")
        return

    counts = {key: sum(1 for v in verdicts.values() if v["verdict"] == key)
              for key in (ANNOTATOR, HUMAN, AMBIGUOUS)}
    print(f"adjudicated {len(verdicts)} of {len(conflicts)} disagreements "
          f"(from {compared} strings judged twice, {agreed} of which agreed outright)\n")
    for key, meaning in ((ANNOTATOR, "guide selects the gold label"),
                         (HUMAN, "guide selects the relabel — gold label is wrong"),
                         (AMBIGUOUS, "the string does not determine one entity")):
        print(f"  {key:10} {counts[key]:3}   {meaning}")

    resolved = counts[ANNOTATOR] + counts[HUMAN]
    print(f"\ngold label defensible on {agreed + counts[ANNOTATOR]} of {compared} "
          f"({100 * (agreed + counts[ANNOTATOR]) / compared:.1f}%) strings judged twice")
    if resolved:
        print(f"of the disagreements that resolve, {100 * counts[ANNOTATOR] / resolved:.1f}% "
              f"go to the gold label")
    print(f"under-determined: {counts[AMBIGUOUS]} of {compared} "
          f"({100 * counts[AMBIGUOUS] / compared:.1f}%) — the ceiling on any single-entity "
          "precision claim")

    wrong = [text for text, v in sorted(verdicts.items()) if v["verdict"] == HUMAN]
    if wrong:
        print(f"\ngold labels the guide rejects ({len(wrong)}) — candidates for correction:")
        for text in wrong:
            record = verdicts[text]
            print(f"  {text!r}\n      gold:      "
                  f"{option_text(record['claude_fdc_id'], description)}"
                  f"\n      should be: {option_text(record['human_fdc_id'], description)}")


def main() -> None:
    if "--report" in sys.argv:
        report()
        return

    con = duckdb.connect(str(DEFAULT_DB), read_only=True)
    path = Path(DEFAULT_OUT) / "adjudication.jsonl"
    try:
        description = descriptions(con)
        kcal = dict(con.execute(
            "SELECT fdc_id, kcal_per_100g FROM silver.usda_foods").fetchall())
        lines = {row["normalized_text"]: row.get("example_lines", []) for row in load_sample()}
        todo = pending(DEFAULT_OUT, description)

        print(f"blind adjudication — {len(todo)} disagreements to settle")
        print("Two labels, order randomized, neither attributed. The question is NOT which you")
        print("prefer — it is which one docs/labeling_guide.md selects for this line:")
        print("  1 least-qualified base form   2 specific only where the line is specific")
        print("  3 no babyfood/restaurant/brand unless named   4 no-match beats a distant relative")
        print("keys: a / b = that one   ? = genuinely ambiguous   k = skip   q = quit\n")

        for position, (text, claude_fdc, human_fdc) in enumerate(todo, start=1):
            first, second = ((claude_fdc, human_fdc) if order_for(text)
                             else (human_fdc, claude_fdc))
            print(f"[{position}/{len(todo)}]  STRING: {text!r}")
            for line in lines.get(text, []):
                print(f"    seen as: {line}")
            print(f"  a  {option_text(first, description, kcal)}")
            print(f"  b  {option_text(second, description, kcal)}")

            while True:
                choice = input("  [a] [b] [?]ambiguous [k]skip [q]uit > ").strip().lower()
                if choice == "q":
                    print(f"\nsaved {len(load_verdicts(path))} verdicts to {path}")
                    return
                if choice == "k":
                    break
                if choice in ("a", "b", "?"):
                    verdict = AMBIGUOUS if choice == "?" else choice
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(
                            make_verdict(text, claude_fdc, human_fdc, verdict)) + "\n")
                    break
                print("  ?")
            print()

        print(f"done — {len(load_verdicts(path))} verdicts in {path}")
        print("summary: uv run python -m pantryiq.er.adjudicate --report")
    except (KeyboardInterrupt, EOFError):
        print(f"\ninterrupted — {len(load_verdicts(path))} verdicts saved")
    finally:
        con.close()


if __name__ == "__main__":
    main()
