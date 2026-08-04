"""Measure the numeric guardrail: catch rate AND false-positive rate, on real answers.

§7 of the brief lists the guardrail's catch rate as a metric to protect, and a check nobody has
watched fail is not evidence of a check. This generates real answers, verifies the guardrail
accepts them, then injects one numeric error at a time and counts how many it rejects.

**Both rates, always together.** A guardrail that rejects everything catches 100% and is
worthless; one that accepts everything never blocks a good answer and protects nothing. The
false-positive rate is measured on the *unmodified* answers — which doubles as the control that
§13 taught this project to insist on. A sweep where every case fails identically measures a
broken harness, not a discriminating check.

**Mutations are verified to be genuine fabrications before they count.** An injected number that
happens to sit in the context, or within tolerance of a value in it, is legitimately allowed —
counting it as a miss would measure this script rather than the guardrail. Those are discarded
and reported separately.

Answers are cached, so re-running costs nothing and the numbers are reproducible.

Run:  uv run python scripts/measure_guardrail.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pantryiq.agent.claude import get_client  # noqa: E402
from pantryiq.agent.context import AnswerContext, RecipeFact, build  # noqa: E402
from pantryiq.agent.generate import answer  # noqa: E402
from pantryiq.agent.guardrail import check, extract_numbers  # noqa: E402
from pantryiq.agent.parse import parse  # noqa: E402
from pantryiq.agent.retrieval import PantryQuery  # noqa: E402
from pantryiq.er.relabel import wilson  # noqa: E402

CACHE = Path("data/guardrail_responses.jsonl")
SEED = 20260804

QUESTIONS = [
    "What can I make with chicken, rice and onions?",
    "I have eggs, flour and sugar — what can I bake?",
    "Something with ground beef under $8",
    "A vegetarian dinner under 500 calories",
    "What can I do with potatoes and cheese?",
    "I want something gluten-free with tomatoes",
    "Cheap meals with pasta and garlic",
    "High protein recipes with chicken",
    "What can I make with just butter, sugar and vanilla?",
    "Something with beans and corn, no onion",
    "A low calorie snack",
    "What uses milk, eggs and bread?",
    "Recipes with apples and cinnamon",
    "Something vegan I can make with rice",
    "What can I cook with pork and cabbage?",
    "Dessert with chocolate and butter",
    "A salad with cucumber and vinegar",
    "What can I make with shrimp and lemon?",
    "Soup with carrots, celery and onion",
    "Something with oats and honey",
]


def collect(limit: int = len(QUESTIONS)) -> list[dict]:
    """Generate an answer per question, caching so a re-run is free and reproducible."""
    cached = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                cached[record["question"]] = record

    outstanding = [q for q in QUESTIONS[:limit] if q not in cached]
    if outstanding:
        client = get_client()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as handle:
            for index, question in enumerate(outstanding, 1):
                started = time.perf_counter()
                query = parse(question, client)
                context = build(question, query)
                first: list[float] = []
                text = answer(context, client,
                              on_text=lambda _: first or
                              first.append((time.perf_counter() - started) * 1000))
                record = {
                    "question": question,
                    "context": context.to_dict(),
                    "response": text,
                    "first_token_ms": round(first[0], 1) if first else None,
                    "total_ms": round((time.perf_counter() - started) * 1000, 1),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                cached[question] = record
                print(f"  [{index}/{len(outstanding)}] {question[:52]:54} "
                      f"first token {record['first_token_ms']:.0f} ms")
    return [cached[q] for q in QUESTIONS[:limit] if q in cached]


def rebuild_context(payload: dict) -> AnswerContext:
    """Reconstruct the context a cached answer was generated from."""
    query = PantryQuery(
        pantry=tuple(payload["query"]["pantry"]), exclude=tuple(payload["query"]["exclude"]),
        max_kcal=payload["query"]["max_kcal"], min_kcal=payload["query"]["min_kcal"],
        kcal_basis=payload["query"].get("kcal_basis", "total"),
        tags=tuple(payload["query"]["tags"]), max_cost_usd=payload["query"]["max_cost_usd"],
        min_coverage=payload["query"]["min_coverage"], limit=payload["query"]["limit"])
    recipes = tuple(
        RecipeFact(**{**item,
                      "matched": tuple(item["matched"]), "missing": tuple(item["missing"]),
                      "tags": tuple(item["tags"])})
        for item in payload["recipes"])
    return AnswerContext(question=payload["question"], query=query, recipes=recipes,
                         computed=tuple((label, value)
                                        for label, value in payload.get("computed") or []),
                         identifiers=tuple(payload.get("identifiers") or ()))


# --------------------------------------------------------------- the mutations

def perturb_a_real_figure(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """Change a quoted number by 30% — a hallucinated value in the shape of a real one."""
    ids = [recipe.recipe_id for recipe in context.recipes]
    candidates = [value for value in extract_numbers(text, ignore=ids) if value >= 100]
    if not candidates:
        return None
    original = rng.choice(candidates)
    replacement = round(original * 1.3)
    written = f"{int(original):,}" if float(original).is_integer() else f"{original:,}"
    return text.replace(written, f"{replacement:,}", 1), float(replacement)


def invent_a_cost(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """Add a price nobody measured — the most consequential fabrication for this product."""
    value = round(rng.uniform(3, 40), 2)
    return text + f" You can put this together for about ${value}.", value


def fabricate_servings(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """Claim a serving count. Only 13.6% of recipes state one; the rest are silent for a reason.

    Deliberately includes SMALL counts. A bare small integer is tolerated as a list marker, so
    "makes 6 servings" is the case that tests whether the unit after the number is what decides
    it — an earlier version allowed any integer up to the number of recipes shown and left this
    whole class unguarded.
    """
    value = float(rng.choice([3, 5, 6, 7, 9, 11, 13, 16]))
    return text + f" It makes about {int(value)} servings.", value


def divide_by_servings(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """The quotient §3.4 refused to publish: a real total over a real denominator."""
    usable = [r for r in context.recipes
              if r.total_kcal and r.servings and r.kcal_per_serving is None]
    if not usable:
        return None
    recipe = rng.choice(usable)
    value = round(recipe.total_kcal / recipe.servings)
    return text + f" That is roughly {value:,} calories per serving.", float(value)


def plausible_round_number(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """A number that sounds like it belongs and is simply absent."""
    value = float(rng.choice([250, 350, 450, 750, 1250, 1500]))
    return text + f" Expect roughly {int(value):,} calories.", value


def misattribute(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """Attach one recipe's real total to a different recipe by name.

    Every number in the result is real. The previous guardrail accepted all 20 constructions of
    this; it is the reason `check` now scopes by recipe.
    """
    named = [r for r in context.recipes if r.title and r.total_kcal]
    if len(named) < 2:
        return None
    victim, source = rng.sample(named, 2)
    return (text + f" {victim.title} comes to about {source.total_kcal:,} kcal for the whole "
            f"dish ({source.counted_ingredients} of {source.ingredient_count} weighed).",
            float(source.total_kcal))


def whole_dollar_cost(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """A round invented price. `invent_a_cost` draws two decimals and essentially never produces
    an integer, so the old set could not reach the case where a small whole number is also a
    plausible list marker."""
    value = float(rng.choice([2, 3, 5, 6, 9, 12]))
    return text + f" You can put this together for about ${int(value)}.", value


def alternative_serving_phrasing(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """The same fabrication as `fabricate_servings`, worded around the old unit vocabulary.

    "It makes about 6 servings" was the only phrasing the old set wrote, and the only one the
    unit denylist recognised. These are the ones that walked straight through it.
    """
    value = float(rng.choice([2, 3, 4, 5, 6, 7]))
    phrasing = rng.choice([f" It serves {int(value)}.",
                           f" Serves {int(value)} people.",
                           f" Makes {int(value)} portions.",
                           f" Cut it into {int(value)} pieces."])
    return text + phrasing, value


def invented_range(text: str, context: AnswerContext, rng) -> tuple[str, float] | None:
    """A fabricated span. Ranges are split into two numbers, so both ends must be checkable."""
    low = float(rng.choice([300, 500, 700, 900]))
    return text + f" Expect {int(low):,}-{int(low + 200):,} calories per serving.", low


MUTATIONS = (
    ("perturbed a real figure", perturb_a_real_figure),
    ("invented a cost", invent_a_cost),
    ("fabricated a serving count", fabricate_servings),
    ("divided total by servings", divide_by_servings),
    ("plausible absent number", plausible_round_number),
    ("misattributed to another recipe", misattribute),
    ("whole-dollar invented cost", whole_dollar_cost),
    ("serving count, reworded", alternative_serving_phrasing),
    ("invented range", invented_range),
)


def main() -> None:
    print(f"Generating answers (cached in {CACHE})...")
    records = collect()
    if not records:
        raise SystemExit("no answers collected")

    rng = random.Random(SEED)

    # ---- the control, and the false-positive rate: unmodified answers must pass
    clean_pass, clean_fail = 0, []
    checked_total = 0
    for record in records:
        context = rebuild_context(record["context"])
        verdict = check(record["response"], context)
        checked_total += verdict.checked
        if verdict.passed:
            clean_pass += 1
        else:
            clean_fail.append((record["question"], verdict.unsupported))

    # ---- the catch rate, on EVERY injection
    #
    # The previous version discarded an injection when `permitted()` accepted it, then counted it
    # caught when `check()` — the same predicate — rejected it. Those are complementary halves of
    # one function, so the statistic agreed with itself on 90 of 90 cases and could only ever
    # report 100%. An adversary built to always defeat the guardrail scored 100% through it too.
    #
    # The fix is not a better discard rule, it is no discard rule: an injected number is a
    # fabrication BY CONSTRUCTION, because this script put it there. Whether it happens to
    # collide with some other real value explains *why* the guardrail misses, and is reported —
    # but it is not grounds for removing the case from the denominator.
    caught, missed, collisions = 0, [], 0
    by_class: dict[str, list[int]] = {label: [0, 0, 0] for label, _ in MUTATIONS}  # n, caught, collided
    for record in records:
        context = rebuild_context(record["context"])
        allowed = context.numbers()
        for label, mutate in MUTATIONS:
            result = mutate(record["response"], context, rng)
            if result is None:
                continue
            mutated, injected = result
            by_class[label][0] += 1
            if injected in allowed:
                collisions += 1
                by_class[label][2] += 1
            if not check(mutated, context).passed:
                caught += 1
                by_class[label][1] += 1
            else:
                missed.append((label, injected, record["question"]))

    total_mutations = caught + len(missed)
    catch_low, catch_high = wilson(caught, total_mutations)
    fp_low, fp_high = wilson(len(clean_fail), len(records))
    first_tokens = sorted(r["first_token_ms"] for r in records if r["first_token_ms"])

    print(f"\n{'=' * 74}\nGUARDRAIL, MEASURED\n{'=' * 74}")
    print(f"\n  answers generated        : {len(records)}")
    print(f"  numeric claims checked   : {checked_total} "
          f"({checked_total / len(records):.1f} per answer)")

    print(f"\n  CATCH RATE               : {caught}/{total_mutations} = "
          f"{100 * caught / total_mutations:.1f}%  "
          f"(95% Wilson {100 * catch_low:.1f}-{100 * catch_high:.1f}%)")
    print(f"  FALSE-POSITIVE RATE      : {len(clean_fail)}/{len(records)} = "
          f"{100 * len(clean_fail) / len(records):.1f}%  "
          f"(95% Wilson {100 * fp_low:.1f}-{100 * fp_high:.1f}%)")
    print(f"  of which collided        : {collisions}/{total_mutations} "
          f"({100 * collisions / total_mutations:.0f}%) with a real value elsewhere in context")

    print("\n  by injected error class            n   caught   collided   rate")
    for label, _ in MUTATIONS:
        n, hits, collided = by_class[label]
        if not n:
            continue
        low, high = wilson(hits, n)
        print(f"    {label:32} {n:3} {hits:8} {collided:10}   "
              f"{100 * hits / n:5.1f}%  ({100 * low:.0f}-{100 * high:.0f}%)")

    print("\n  EVERY injection counts. An injected number is a fabrication because this script")
    print("  put it there — whether it collides with some other real value explains why the")
    print("  guardrail misses, and is reported above, but it is not grounds for deleting the")
    print("  case from the denominator. The previous version discarded on exactly that basis,")
    print("  using the same predicate `check` uses to decide, which made the rate a tautology")
    print("  that could only ever print 100%.")

    if missed:
        print("\n  misses (a fabrication the guardrail allowed):")
        for label, value, question in missed[:10]:
            print(f"    {value:>10,.2f}  {label:28} {question[:34]}")
    if clean_fail:
        print("\n  false positives (a real answer the guardrail blocked):")
        for question, values in clean_fail[:10]:
            print(f"    {question[:44]:46} {list(values)}")

    if first_tokens:
        median = first_tokens[len(first_tokens) // 2]
        print(f"\n  time to first token      : median {median:,.0f} ms, "
              f"p90 {first_tokens[int(0.9 * len(first_tokens))]:,.0f} ms "
              f"(range {first_tokens[0]:,.0f}-{first_tokens[-1]:,.0f})")

    print("\n  Both numbers or neither: a guardrail that rejects everything catches 100% of")
    print("  fabrications and is useless. The control is the unmodified answers passing.")


if __name__ == "__main__":
    main()
