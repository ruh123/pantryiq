"""Blind relabeling — turn the label-provenance caveat into a measured number.

282 of the 300 gold labels came from an LLM annotator (see `docs/er_metrics.md`), so every
metric built on them measures *pipeline-vs-annotator agreement*, not accuracy. This draws a
frozen random subset of those strings, has a human judge them **without seeing the existing
label**, and reports the human/LLM agreement rate.

Four properties keep that rate honest:

- **The draw is seeded and frozen** into `relabel_manifest.json` before any judging starts, so
  it cannot be quietly re-rolled until it yields a flattering number. The manifest records the
  seed and a fingerprint of the population it was drawn from, so the draw is reproducible by
  anyone who doubts it.
- **It is drawn uniformly from all claude-labeled strings**, not from the 85 low-confidence
  ones. Those are the best targets for *fixing* labels, but sampling them would measure a
  biased-pessimistic rate rather than one that generalizes to the label set.
- **The original label is never shown.** `labels.jsonl` is read once, when the manifest is
  first written, to identify the population; the judging loop afterwards reads only the
  manifest and `relabel.jsonl`.
- **Judgments land in a separate file.** `labels.jsonl` is never mutated. Whether to promote a
  human correction into the gold set is a separate decision, taken after seeing the rate.

Agreement credits any entity sharing the original's `description_raw`: 94 USDA descriptions
exist twice (Foundation + SR Legacy), and picking the other twin is not a disagreement — the
same convention `docs/labeling_guide.md` states for scoring.

Run:  uv run python -m pantryiq.er.relabel            # judge the drawn strings
      uv run python -m pantryiq.er.relabel --report   # agreement rate + disagreements
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import duckdb

from pantryiq.er.gold import DEFAULT_OUT
from pantryiq.er.labeling import (
    DEFAULT_DB,
    NO_MATCH,
    label_rows,
    load_labels,
    load_sample,
    verify_sample,
)

# Frozen on purpose, and deliberately not exposed as CLI flags: a re-rollable seed or sample
# size is a re-rollable result.
SEED = 20260729
SAMPLE_SIZE = 30
ANNOTATOR = "claude"


def population(labels: dict[str, dict], labeler: str = ANNOTATOR) -> list[str]:
    """The strings eligible for relabeling, in a deterministic order."""
    return sorted(text for text, record in labels.items() if record.get("labeler") == labeler)


def fingerprint(strings: list[str]) -> str:
    return hashlib.sha256("\n".join(strings).encode()).hexdigest()[:16]


def draw(labels: dict[str, dict], size: int = SAMPLE_SIZE, seed: int = SEED) -> list[str]:
    """A seeded uniform draw from the annotator-labeled strings.

    Sorting the population first makes the draw independent of the order labels happen to sit
    in the file, so it reproduces from the seed alone.
    """
    eligible = population(labels)
    return sorted(random.Random(seed).sample(eligible, min(size, len(eligible))))


def ensure_manifest(out_dir: Path | str = DEFAULT_OUT) -> dict:
    """Draw once, then never again — an existing manifest is authoritative."""
    path = Path(out_dir) / "relabel_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    labels = load_labels(Path(out_dir) / "labels.jsonl")
    eligible = population(labels)
    strings = draw(labels)
    manifest = {
        "seed": SEED,
        "size": len(strings),
        "drawn_from": f"{ANNOTATOR}-labeled strings",
        "population_size": len(eligible),
        "population_fingerprint": fingerprint(eligible),
        "strings": strings,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def rows_to_judge(sample: list[dict], manifest: dict, done: dict[str, dict]) -> list[dict]:
    """Sample rows for drawn strings not yet relabeled — carrying no gold label with them."""
    drawn = [text for text in manifest["strings"] if text not in done]
    by_text = {row["normalized_text"]: row for row in sample}
    return [by_text[text] for text in drawn if text in by_text]


def same_entity(a: str, b: str, description: dict[str, str]) -> bool:
    """Two judgments agree if they name the same entity — or the same description."""
    if a == b:
        return True
    if a == NO_MATCH or b == NO_MATCH:
        return False
    first, second = description.get(a), description.get(b)
    return first is not None and first == second


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — the right interval for a proportion at n=30.

    The normal approximation misbehaves near 0 and 1, which is exactly where a good agreement
    rate lands; Wilson stays inside [0, 1] and does not collapse to zero width at 100%.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def agreement(original: dict[str, dict], relabeled: dict[str, dict],
              description: dict[str, str]) -> tuple[int, int, list[tuple[str, str, str]]]:
    """(agreements, compared, disagreements) over the strings judged twice."""
    agreed = compared = 0
    disagreements = []
    for text, record in sorted(relabeled.items()):
        if text not in original:
            continue
        compared += 1
        theirs, ours = original[text]["fdc_id"], record["fdc_id"]
        if same_entity(theirs, ours, description):
            agreed += 1
        else:
            disagreements.append((text, theirs, ours))
    return agreed, compared, disagreements


def descriptions(con) -> dict[str, str]:
    return {
        row[0]: row[1]
        for row in con.execute("SELECT fdc_id, description_raw FROM silver.usda_foods").fetchall()
    }


def _describe(fdc_id: str, description: dict[str, str]) -> str:
    return NO_MATCH if fdc_id == NO_MATCH else f"{description.get(fdc_id, '?')} ({fdc_id})"


def report(out_dir: Path | str = DEFAULT_OUT, db_path: Path | str = DEFAULT_DB) -> None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        description = descriptions(con)
    finally:
        con.close()

    original = load_labels(Path(out_dir) / "labels.jsonl")
    relabeled = load_labels(Path(out_dir) / "relabel.jsonl")
    manifest = json.loads((Path(out_dir) / "relabel_manifest.json").read_text(encoding="utf-8"))
    agreed, compared, disagreements = agreement(original, relabeled, description)

    if not compared:
        print("no relabeled strings yet — run `uv run python -m pantryiq.er.relabel`")
        return

    low, high = wilson(agreed, compared)
    print(f"blind relabel: {compared} of {manifest['size']} drawn strings judged twice")
    print(f"human/{ANNOTATOR} agreement: {agreed}/{compared} = {100 * agreed / compared:.1f}% "
          f"(95% Wilson CI {100 * low:.1f}–{100 * high:.1f}%)")

    for flag in ("high", "low"):
        subset = [t for t in relabeled if original.get(t, {}).get("confidence") == flag]
        if subset:
            hits = sum(1 for t in subset
                       if same_entity(original[t]["fdc_id"], relabeled[t]["fdc_id"], description))
            print(f"  original confidence={flag:4}  n={len(subset):3}  "
                  f"{100 * hits / len(subset):5.1f}%")

    if disagreements:
        print(f"\n{len(disagreements)} disagreements ({ANNOTATOR} → human)")
        for text, theirs, ours in disagreements:
            print(f"  {text!r}\n      {ANNOTATOR}: {_describe(theirs, description)}"
                  f"\n      human:  {_describe(ours, description)}")


def main() -> None:
    if "--report" in sys.argv:
        report()
        return

    con = duckdb.connect(str(DEFAULT_DB), read_only=True)
    relabel_path = Path(DEFAULT_OUT) / "relabel.jsonl"
    try:
        sample = load_sample()
        verify_sample(sample, con)
        manifest = ensure_manifest()
        done = load_labels(relabel_path)
        todo = rows_to_judge(sample, manifest, done)

        print(f"blind relabel — {len(done)} of {manifest['size']} judged, {len(todo)} to go")
        print("You are labeling these fresh. The existing label is NOT shown, and nothing you")
        print("do here touches labels.jsonl — this measures agreement, it does not overwrite.")
        print("keys: ENTER = accept the top row   number = pick   n = no-match")
        print("      s = search all foods   m = show more   k = skip   q = quit")
        print("rules: docs/labeling_guide.md   (~ = deprioritized: babyfood/restaurant/brand)\n")

        if label_rows(con, todo, relabel_path, extra={"labeler": "human", "pass": "blind"}):
            return
        print(f"done — {len(load_labels(relabel_path))} judgments in {relabel_path}")
        print("agreement: uv run python -m pantryiq.er.relabel --report")
    except (KeyboardInterrupt, EOFError):
        print(f"\ninterrupted — {len(load_labels(relabel_path))} judgments saved")
    finally:
        con.close()


if __name__ == "__main__":
    main()
