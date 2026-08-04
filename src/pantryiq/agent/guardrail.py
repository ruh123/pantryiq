"""The numeric guardrail — deterministic code that decides whether an answer may be shown.

Every number in a generated response must trace to the context, **and to the right part of it**.
Anything else is a fabrication, however plausible it reads, and this rejects it.

**Deterministic, not a second model.** An LLM checking an LLM shares the failure mode of the
thing it is checking, and this project exists to replace that arrangement with measurement.

---

## What the first version got wrong

It was published claiming a 100% catch rate. That number was a **tautology**: the measurement
harness discarded an injection when `permitted()` accepted it, then counted it caught when
`check()` — the same predicate — rejected it. The two were complementary halves of one function,
agreeing on 90 of 90 injections. On the honest denominator it caught **40%**.

Three causes, all fixed here:

1. **The union was the scope.** A number was accepted if it appeared *anywhere* in the context,
   so an answer could quote one recipe's calories while naming another — median 133% wrong, up to
   1,105%, and in 7 of 20 contexts it also upgraded an incomplete recipe to "6 of 6 ingredients
   weighed", falsifying the coverage guarantee that is the project's central honesty claim.
   Now the response is **segmented by which recipe it is talking about**, and each number is
   checked against that recipe plus the answer-level facts.

2. **The tolerance was borrowed from a food-equivalence constant.** `er/nutrition.EQUIVALENT`
   means "two USDA foods are nutritionally interchangeable" — a category error when reused as
   quoting accuracy. Symmetric on the max, it made every allowed value claim a 21%-wide interval;
   over ~81 values that covered **79% of the number line**. It was needed by **0 of 346** real
   numeric claims: 345 were exact and 1 was a list marker. `QUOTE_TOLERANCE` below is this
   module's own, and small, because `context.py` shows the model display-rounded values in the
   first place.

3. **Counts and measurements were split by magnitude.** `EXACT_BELOW = 20` treated every small
   number as a count, rejecting "about 13 g of protein" against a stored 13.1 — 51% of rounded
   sub-20g macro quotations. The split is now by *kind*: `RecipeFact.exact_numbers()` are counts
   and get no tolerance at any magnitude; `measured_numbers()` were weighed and get the band.

**List markers are recognised by position, not by vocabulary.** The old rule allowed any small
integer unless a unit word followed it, from a fixed list — the same denylist anti-pattern this
project condemns in the dietary tags, and it let through "It serves 2", "Makes 2 portions" and
"$2 total". A bare integer is now tolerated only where it actually is a list marker: at the start
of a line, followed by `.` or `)`.

**Quotients and sums remain forbidden.** A recipe can carry `servings: 8` and `kcal_per_serving:
unknown` at once, because §3.4 refused to publish that quotient below full coverage. 3,470/8 =
433.75 is exactly the number that refusal exists to prevent.

**A failure degrades, it does not crash.** Mismatch -> regenerate once with the offending number
named -> still failing -> a templated answer built directly from the context.

Run:  uv run python -m pantryiq.agent.guardrail
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from pantryiq.agent.context import AnswerContext, RecipeFact

# This module's own tolerance, deliberately not `er/nutrition.EQUIVALENT` (0.10, "nutritionally
# interchangeable"). 1% covers display rounding — "about 1,970" for 1,968 — and nothing else.
# Measured cost of the old 10%: it accepted 79% of the number line; 0 of 346 real claims needed
# any band at all.
QUOTE_TOLERANCE = 0.01

# Numbers as they appear in prose: 1,970 / 11.05 / .5 / 80.
_NUMBER = re.compile(r"\d[\d,]*\.?\d*|\.\d+")
# Struck after identifier removal so "1,970-2,300" yields two numbers and not "1,9702,300".
_RANGE_SEPARATORS = re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)")
# A list marker: an integer opening a line, followed by a dot or bracket. Positional, so it
# cannot be widened by phrasing the way a unit vocabulary could.
_LIST_MARKER = re.compile(r"(?m)^[\s>*-]*(\d{1,2})[.)]\s")
# Sentence boundaries, kept in the split so offsets stay aligned with the original string. A
# decimal like "40.8" is safe: the dot is followed by a digit, not whitespace.
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _sentences(text: str) -> list[tuple[int, str]]:
    """(offset, sentence) covering the whole string, so mention positions stay comparable."""
    pieces, last = [], 0
    for match in _SENTENCE.finditer(text):
        pieces.append((last, text[last:match.end()]))
        last = match.end()
    pieces.append((last, text[last:]))
    return pieces


def _strippable(identifier: str) -> bool:
    """Whether an identifier may be removed before scanning.

    A purely numeric identifier would delete real quantities from the response: with
    `identifiers=("9999",)`, the sentence "9999 rows were dropped" became invisible to the
    guardrail. Untrusted text reaches this channel from `explain.py`, so it is checked rather
    than assumed well-formed.
    """
    return bool(identifier) and any(not character.isdigit() for character in identifier)


def _scan(text: str, ignore: Iterable[str] = ()) -> list[tuple[float, bool]]:
    """(value, is_list_marker) per numeric token.

    Identifiers are stripped **before** range-splitting: doing it the other way round rewrote
    `Chocolate, dark, 60-69% cacao` into a form the literal `str.replace` could no longer find,
    leaving 60 and 69 in the response as unexplained numbers.
    """
    cleaned = text
    for identifier in sorted((i for i in ignore if _strippable(i)), key=len, reverse=True):
        cleaned = cleaned.replace(identifier, " ")
    marker_spans = {match.start(1) for match in _LIST_MARKER.finditer(cleaned)}
    cleaned = _RANGE_SEPARATORS.sub(" ", cleaned)

    values = []
    for match in _NUMBER.finditer(cleaned):
        token = match.group().replace(",", "").rstrip(".")
        if token and token != ".":
            values.append((float(token), match.start() in marker_spans))
    return values


def extract_numbers(text: str, ignore: Iterable[str] = ()) -> list[float]:
    """Every numeric token in the response, in order of appearance."""
    return [value for value, _ in _scan(text, ignore)]


def permitted(value: float, exact: set[float], measured: set[float],
              is_marker: bool = False) -> bool:
    """Whether one number is entitled to appear in the scope it was found in.

    Exact values admit no tolerance at any magnitude — a count is right or it is wrong.
    Measurements admit `QUOTE_TOLERANCE`, which exists only for display rounding.
    """
    if value in exact or value in measured:
        return True
    if is_marker:
        return True
    return any(abs(value - candidate) <= QUOTE_TOLERANCE * max(abs(candidate), 1.0)
               for candidate in measured)


def _needles(recipe: RecipeFact) -> list[str]:
    """The spellings of a recipe's name an answer might actually use.

    The corpus writes titles like `Refrigerator Mashed Potatoes(Serves 12)` and `Reeses
    Cups(Candy)`; a model naturally drops the parenthetical, and then the title never matches and
    that recipe's whole section is attributed to whichever recipe was named before it.
    """
    title = (recipe.title or "").strip()
    # Split on the FIRST "(" rather than stripping a trailing group: the corpus nests them, as in
    # `Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies)`, and `\([^)]*\)$` cannot match that.
    # When it failed, the whole recipe's section was attributed to the recipe named before it.
    trimmed = title.split("(")[0].strip()
    return [needle for needle in dict.fromkeys([title, trimmed, recipe.recipe_id])
            if needle and len(needle) >= 3]


def mentions(response: str, context: AnswerContext) -> list[tuple[int, RecipeFact]]:
    """Where each recipe is named, with overlaps resolved in favour of the longest name."""
    found: list[tuple[int, int, RecipeFact]] = []
    for recipe in context.recipes:
        for needle in _needles(recipe):
            for match in re.finditer(re.escape(needle), response, re.IGNORECASE):
                found.append((match.start(), match.end(), recipe))

    # "Potato Casserole" is a substring of "Hash Brown Potato Casserole"; without this the
    # segmenter cut one recipe's section in two and checked the first recipe's calories against
    # the second's values — a false positive on a correct answer.
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, RecipeFact]] = []
    covered_to = -1
    for start, end, recipe in found:
        if start < covered_to:
            continue
        kept.append((start, recipe))
        covered_to = end
    return kept


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
                "you were given, or belongs to a different recipe than the one you attached it "
                "to. Every number must appear in the data for the recipe you are describing. Do "
                "not compute new figures — in particular, do not divide a total by a serving "
                "count. Rewrite the answer using only the numbers shown.")


def check(response: str, context: AnswerContext) -> Verdict:
    """Verify every number in `response` against the recipe it is attached to.

    Scope is decided **per sentence**, not per section. A sentence that names two recipes puts
    both in scope — "X and Y are also options, at 1,947 and 2,731 kcal respectively" is a correct
    sentence, and section-level scoping rejected it. A sentence naming none inherits the last
    recipe named before it, which is what carries a bulleted recipe's follow-on lines.
    """
    # Titles are identifiers too: `Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies)` puts a
    # 22 and a 2 into the answer as soon as the model names the dish, and those digits are the
    # recipe's name rather than a measurement of it.
    identifiers = ([r.recipe_id for r in context.recipes]
                   + [r.title for r in context.recipes if r.title]
                   + list(context.identifiers))
    global_exact = context.global_exact()
    global_measured = context.global_measured()
    everything_exact = set(global_exact)
    everything_measured = set(global_measured)
    for recipe in context.recipes:
        everything_exact |= recipe.exact_numbers()
        everything_measured |= recipe.measured_numbers()

    named_at = mentions(response, context)
    unsupported: list[float] = []
    checked = 0
    current: RecipeFact | None = None

    for offset, piece in _sentences(response):
        span = range(offset, offset + len(piece))
        here = [recipe for start, recipe in named_at if start in span]
        if here:
            current = here[-1]
        scope = here or ([current] if current else [])

        if scope:
            exact = set(global_exact)
            measured = set(global_measured)
            for recipe in scope:
                exact |= recipe.exact_numbers()
                measured |= recipe.measured_numbers()
        else:
            # Before any recipe is named: an opening summary may reference any of them.
            exact, measured = everything_exact, everything_measured

        for value, is_marker in _scan(piece, ignore=identifiers):
            checked += 1
            if not permitted(value, exact, measured, is_marker):
                unsupported.append(value)


    return Verdict(passed=not unsupported, checked=checked,
                   unsupported=tuple(dict.fromkeys(unsupported)))


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

    The returned verdict always describes the returned text. An earlier version paired the
    templated fallback with the *rejected retry's* verdict, so `log.record` stored a clean answer
    alongside `guardrail_pass=False` and numbers that appeared nowhere in it — corrupting the
    running catch rate on exactly the rows where the guardrail had fired.

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

    fallback = templated_answer(context)
    return fallback, check(fallback, context), True


def main() -> None:
    from pantryiq.agent.context import build
    from pantryiq.agent.retrieval import PantryQuery

    context = build("What can I make with chicken, rice and onions?",
                    PantryQuery(pantry=("chicken", "rice", "onion"), limit=3))
    top, other = context.recipes[0], context.recipes[1]

    honest = (f"{top.title} uses all three, about {top.total_kcal:,} kcal for the whole dish "
              f"({top.counted_ingredients} of {top.ingredient_count} ingredients weighed).")
    cases = [
        ("honest", honest),
        ("invented total", honest.replace(f"{top.total_kcal:,}", "2,900")),
        ("divided by servings",
         honest + f" That is roughly {top.total_kcal / (top.servings or 8):,.0f} a serving."),
        # The failure the union scope could not see: every number real, attached to the wrong dish.
        ("misattributed",
         f"{top.title} is your best match — about {other.total_kcal:,} kcal for the whole dish "
         f"({other.counted_ingredients} of {other.ingredient_count} ingredients weighed)."),
        ("bare integer with a unit", honest + " It makes about 2 servings."),
        ("list marker", "1. " + honest),
    ]
    for label, text in cases:
        verdict = check(text, context)
        mark = "PASS" if verdict.passed else "FAIL"
        print(f"  {mark}  {label:26} {verdict.checked} checked"
              f"{'' if verdict.passed else '  unsupported: ' + str(list(verdict.unsupported))}")

    print("\n  'misattributed' is the one the previous version accepted: every figure in it is")
    print("  real, and all of them belong to a different recipe than the one being named.")


if __name__ == "__main__":
    main()
