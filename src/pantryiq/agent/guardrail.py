"""The numeric guardrail — deterministic code that decides whether an answer may be shown.

Every number in a generated response must trace to the context. Anything else is a fabrication,
however plausible it reads, and this rejects it. §7 of the brief lists the catch rate as a metric
to protect, so `scripts/measure_guardrail.py` measures it against injected errors rather than
asserting it holds.

**Deterministic, not a second model.** An LLM checking an LLM shares the failure mode of the
thing it is checking, and this project exists to replace that arrangement with measurement.

**What counts as permitted, and why the list is short.** A number passes if it matches a context
value within `EQUIVALENT` tolerance, or is an ordinal within the number of recipes shown (a
model writes "1." and "2." in a list). That is deliberately *all* — in particular:

- **Quotients are not permitted.** A recipe can carry `servings: 8` and `kcal_per_serving:
  unknown` at once, because §3.4 refused to publish per-serving figures below full coverage.
  3,470 / 8 = 433.75 is exactly the number that refusal exists to prevent, and admitting
  division as a derivation would sanction it.
- **Sums are not permitted either, and that was measured rather than assumed.** Allowing subset
  sums over 8 recipes admits 256 extra values per field — a large widening of what passes, paid
  for only if real answers actually need it. They did not: measured false-positive rate on
  unmodified answers is reported by `scripts/measure_guardrail.py`, and the rule stays this tight
  until an answer is observed failing for a legitimate sum.

**Tolerance reuses `er/nutrition.error`** — a relative gap with an absolute floor — so "within
10%" means one thing across the project. The floor matters here: a model writing "about 1,970"
for 1,968 is quoting, not inventing.

**A failure degrades, it does not crash.** Mismatch -> regenerate once with the offending number
named -> still failing -> a templated answer built directly from the context. A guardrail failure
costs the user a plainer answer, never a wrong one.

Run:  uv run python -m pantryiq.agent.guardrail
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from pantryiq.agent.context import AnswerContext
from pantryiq.er.nutrition import EQUIVALENT, error

# Numbers as they appear in prose: 1,970 / 11.05 / .5 / 80 — with separators, without a sign.
# A leading minus is deliberately not consumed: no quantity in the context is negative, so "-5"
# would be a fabrication and must be extracted as 5 rather than skipped.
_NUMBER = re.compile(r"\d[\d,]*\.?\d*|\.\d+")
# Struck first so "1,970-2,300" yields two numbers and not "1,9702,300".
_RANGE_SEPARATORS = re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)")
# Below this, a relative tolerance is meaningless — an answer saying "6 ingredients" against a
# context value of 5 must fail, and 10% of 5 would let it pass.
EXACT_BELOW = 20.0
# A unit immediately after a number makes it a measurement rather than an enumeration.
_UNIT = re.compile(r"\s*(servings?|kcal|calor|gram|grams|g\b|ingredient|%|percent|cup|tbsp|tsp|"
                   r"ounce|oz\b|pound|lb\b|minute|hour|dollar)", re.IGNORECASE)


def extract_numbers(text: str, ignore: Iterable[str] = ()) -> list[float]:
    """Every numeric token in the response, in order of appearance.

    Ordinary prose punctuation is handled rather than assumed away: thousands separators,
    currency and percent signs, and en-dash ranges all appear in real answers.

    `ignore` removes identifiers before extraction. Both false positives in the first measured
    run were recipe ids — a model citing `recipenlg:14797` is quoting the context exactly, and
    reading the digits inside an identifier as a quantity blocked two correct answers. Only ids
    actually present in the context are stripped, so an invented id is still caught.
    """
    return [value for value, _ in _scan(text, ignore)]


def _scan(text: str, ignore: Iterable[str] = ()) -> list[tuple[float, bool]]:
    """(value, is_quantified) per numeric token — quantified meaning a unit follows it.

    The distinction closes a hole the ordinal allowance would otherwise open. "1." and "3" are
    enumeration and have to be tolerated; "3 servings" and "6 ingredients" are measurements, and
    letting a small integer through as an ordinal would leave exactly the claim §3.4 is most
    careful about — a serving count — unguarded whenever it happened to be small.
    """
    cleaned = _RANGE_SEPARATORS.sub(" ", text)
    for identifier in ignore:
        cleaned = cleaned.replace(identifier, " ")
    values = []
    for match in _NUMBER.finditer(cleaned):
        token = match.group().replace(",", "").rstrip(".")
        if token and token != ".":
            values.append((float(token),
                           bool(_UNIT.match(cleaned, match.end()))))
    return values


def permitted(value: float, allowed: set[float], ordinal_limit: int,
              quantified: bool = False) -> bool:
    """Whether one number is entitled to appear.

    Small values are compared exactly. A 10% band around 5 spans 4.5 to 5.5, which would let an
    answer claim 6 of 5 ingredients were weighed; counts have to be right, not close.
    """
    if value in allowed:
        return True
    if not quantified and value == int(value) and 1 <= value <= ordinal_limit:
        return True  # "1.", "2." — list markers, bounded by how many recipes were shown
    if value < EXACT_BELOW:
        return False
    return any(error(value, candidate, floor=0.0) < EQUIVALENT for candidate in allowed
               if candidate >= EXACT_BELOW)


@dataclass(frozen=True)
class Verdict:
    """The guardrail's decision, with everything needed to explain or log it."""

    passed: bool
    checked: int
    unsupported: tuple[float, ...]

    def complaint(self) -> str:
        """The correction handed back to the model on a regeneration."""
        listed = ", ".join(f"{value:,g}" for value in self.unsupported)
        verb = "is" if len(self.unsupported) == 1 else "are"
        return (f"Your previous answer contained {listed}, which {verb} not in the recipe data "
                "you were given. "
                "Every number must appear in the data verbatim. Do not compute new figures — "
                "in particular, do not divide a total by a serving count. Rewrite the answer "
                "using only the numbers shown.")


def check(response: str, context: AnswerContext) -> Verdict:
    """Verify every number in `response` against `context`."""
    allowed = context.numbers()
    limit = max(len(context.recipes), 1)
    # Longest first: `kcal_per_100g` must be stripped before a shorter name that is a prefix of
    # it, or the leftover digits read as a claim.
    names = sorted([r.recipe_id for r in context.recipes] + list(context.identifiers),
                   key=len, reverse=True)
    scanned = _scan(response, ignore=names)
    unsupported = tuple(value for value, quantified in scanned
                        if not permitted(value, allowed, limit, quantified))
    return Verdict(passed=not unsupported, checked=len(scanned), unsupported=unsupported)


def templated_answer(context: AnswerContext) -> str:
    """The fallback when generation cannot produce a checkable answer twice running.

    Built from the context directly, so it is correct by construction and needs no checking. It
    is plainer than a written answer, which is the right way for this failure to land.
    """
    if not context.recipes:
        return ("Nothing in the warehouse matches that. Try relaxing one of the constraints — "
                "fewer ingredients, or a wider calorie or cost range.")

    lines = [f"Here is what the data shows for: {context.question}", ""]
    for recipe in context.recipes:
        parts = [f"- {recipe.title or recipe.recipe_id}"]
        if recipe.matched:
            parts.append(f"uses {', '.join(recipe.matched)}")
        if recipe.total_kcal is not None:
            parts.append(f"{recipe.total_kcal:,} kcal for the whole dish "
                         f"({recipe.counted_ingredients} of {recipe.ingredient_count} "
                         "ingredients weighed)")
        if recipe.kcal_per_serving is not None:
            parts.append(f"{recipe.kcal_per_serving:,} kcal per serving")
        if recipe.cost_total_usd is not None:
            qualifier = "" if recipe.cost_coverage == 1.0 else " or more, partly priced"
            parts.append(f"${recipe.cost_total_usd:,}{qualifier}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def guarded_answer(context: AnswerContext, client=None, on_text=None) -> tuple[str, Verdict, bool]:
    """Generate, check, regenerate once, then fall back. Returns (answer, verdict, regenerated).

    Nothing is streamed to the user here: the guardrail needs a complete response before any of
    it can be trusted, so `on_text` is for measurement, not display.
    """
    from pantryiq.agent.generate import answer

    text = answer(context, client, on_text=on_text)
    verdict = check(text, context)
    if verdict.passed:
        return text, verdict, False

    retried = answer(context, client, note=verdict.complaint())
    second = check(retried, context)
    if second.passed:
        return retried, second, True
    return templated_answer(context), second, True


def main() -> None:
    from pantryiq.agent.context import build
    from pantryiq.agent.retrieval import PantryQuery

    context = build("What can I make with chicken, rice and onions?",
                    PantryQuery(pantry=("chicken", "rice", "onion"), limit=3))
    top = context.recipes[0]

    honest = (f"{top.title} uses all three, about {top.total_kcal:,} kcal for the whole dish "
              f"({top.counted_ingredients} of {top.ingredient_count} ingredients weighed).")
    invented = honest.replace(f"{top.total_kcal:,}", "2,900")
    divided = honest + f" That works out to {top.total_kcal / (top.servings or 8):,.0f} kcal a serving."

    for label, text in (("honest", honest), ("invented total", invented), ("divided", divided)):
        verdict = check(text, context)
        mark = "PASS" if verdict.passed else "FAIL"
        print(f"  {mark}  {label:16} {verdict.checked} numbers checked"
              f"{'' if verdict.passed else '  unsupported: ' + str(list(verdict.unsupported))}")
    print("\n  The third is the one that matters: every number in it is real except the quotient,")
    print("  and that quotient is exactly the figure Phase 3 refused to publish.")


if __name__ == "__main__":
    main()
