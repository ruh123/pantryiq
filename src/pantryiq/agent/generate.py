"""Constrained generation — Claude writes the answer, and sees only the assembled context.

No corpus access, no retrieval control, no tools. The model's entire view of the world is the
`<recipe>` elements `context.to_prompt()` produced, which is what lets 4.5 check the result
mechanically: a number in the response that is not in `AnswerContext.numbers()` came from the
model rather than the warehouse, and that is a defect regardless of how plausible it looks.

**The rules the system prompt enforces are the project's, not stylistic preferences.** Each one
exists because the warehouse can be honestly misread:

- *Never state a nutrition figure without its coverage.* `total_kcal` on a recipe at 0.8 coverage
  is a sum over four fifths of the ingredients. Quoting it bare claims more than Gold supports.
- *Never divide.* `servings` is often known while `kcal_per_serving` is deliberately null; §3.4
  refused to publish that quotient and the answer must not reintroduce it.
- *Never fill a gap from your own knowledge.* A model knows roughly what a casserole costs. That
  knowledge is exactly what this project exists to replace with measurement.
- *Say when nothing matched.* An empty context must produce "nothing in the warehouse matches",
  never a recipe recalled from training.

**`fallbacks="default"` is on.** Opus 5's classifiers can decline with HTTP 200 and an empty
`content` array; the fallback re-runs the request server-side rather than failing the turn. The
whole chain declining still raises `Refused` — see `claude.text_of`.

Run:  uv run python -m pantryiq.agent.generate
"""
from __future__ import annotations

from pantryiq.agent.claude import (
    FALLBACK_BETA,
    MODEL,
    get_client,
    injection_paragraph,
    text_of,
)
from pantryiq.agent.context import AnswerContext

# Thinking is off, and the answer is a short paragraph. `max_tokens` caps thinking plus response
# on Opus 5, so this is headroom rather than a target.
MAX_TOKENS = 2000

SYSTEM = (
    "You are PantryIQ's recipe assistant. You answer strictly from a verified nutrition "
    "warehouse, and the recipes below are the only ones you know about.\n\n"
    "Rules, in order of importance:\n"
    "1. Every number you write must appear verbatim in the context. Do not add, average, "
    "convert, or divide. If a figure is not there, say it is not available.\n"
    "1b. An `<already_computed>` block holds totals worked out and verified in code before you "
    "were called. Quote those freely — they are results, not arithmetic for you to redo — and "
    "describe what they show rather than declining to sum.\n"
    "2. `servings` being known does NOT let you compute calories per serving. When "
    "`kcal_per_serving` is `unknown`, the per-serving figure is unavailable — say so rather "
    "than dividing the total.\n"
    "3. Every nutrition figure must be quoted with its coverage, because a recipe below full "
    "coverage has ingredients nobody could weigh. Write it plainly: 'about 1,970 kcal for the "
    "whole dish (all 6 ingredients weighed)' or '(4 of 5 ingredients weighed, so this is an "
    "undercount)'.\n"
    "4. Never supply a fact the context does not have — not a cost, not a cooking time, not an "
    "ingredient, not a substitution. You do not know anything about these recipes beyond what "
    "is shown.\n"
    "5. If there are no recipes, say plainly that nothing in the warehouse matches, and suggest "
    "relaxing a constraint. Never fall back on a recipe you know from elsewhere.\n"
    "5b. The cooking method IS shown to the user, on the page, beside your answer — you simply "
    "cannot see it, because a method's oven temperatures and timings must not become numbers you "
    "are allowed to quote. Never tell them the method or the cooking times are unavailable. If "
    "they ask how to cook it, point them at the steps shown with each recipe.\n"
    "6. Say why the top recipe ranked first. When the user listed ingredients, it is that it "
    "uses more of them. When they asked for a dish by name, every result matches that name and "
    "the order is how completely the recipe could be verified — say that instead, and never "
    "claim a pantry match nobody made. That is the actual ranking rule and the user is entitled "
    "to it.\n"
    "7. `cost_total_usd` is a floor when `cost_coverage` is below 1.0: only some ingredients "
    "are priced. Never present a partial cost as the price of the dish.\n\n"
    "Be brief and concrete — a short paragraph, or a few recipes with one or two lines each. "
    "Write for a home cook, not a data analyst.\n\n"
    + injection_paragraph("recipes", "recipe data") + "\n\n"
    # The question is fenced on the parse call but was not on this one — the same untrusted text,
    # re-presented unprotected to the model that writes the answer.
    + injection_paragraph("user_question", "question")
)


def answer(context: AnswerContext, client=None, note: str = "", on_text=None) -> str:
    """One Claude call: the assembled context in, prose out.

    **Streamed**, and that is a product decision rather than a technical one. Two sequential
    model calls cost 7-12 s end to end, and neither effort nor thinking has any slack left — but
    the parse is the only step that must finish before anything can be shown, so streaming the
    answer puts the first words on screen in roughly parse + 1 s while the rest arrives as it is
    written. The measured target for this phase is therefore **time to first token**, not time to
    the final character; `main()` reports both.

    The full text is still returned, because the guardrail checks a complete response — nothing
    is shown to a user until 4.5 has passed it. `on_text` exists for the CLI and for measuring
    first-token latency.

    `note` is appended to the user turn when regenerating after a guardrail failure, so the
    second attempt is told what was wrong with the first (see 4.5).
    """
    client = client or get_client()
    user = (f"<user_question>{context.question}</user_question>\n\n"
            f"<recipes>\n{context.to_prompt()}\n</recipes>")
    if note:
        user += f"\n\n<correction>{note}</correction>"

    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        thinking={"type": "disabled"},  # accepted at effort 'high' or lower; this is 'low'
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low"},
    ) as stream:
        for chunk in stream.text_stream:
            if on_text:
                on_text(chunk)
        final = stream.get_final_message()

    # Checked on the final message, so a mid-stream classifier decline is caught even though
    # partial text already arrived.
    return text_of(final).strip()


def main() -> None:
    import time

    from pantryiq.agent.context import build
    from pantryiq.agent.parse import parse

    questions = [
        "What can I make with chicken, rice and onions?",
        "I want a vegetarian dinner under 500 calories",
        "What can I cook with unobtainium and moon cheese?",
    ]
    client = get_client()
    for question in questions:
        started = time.perf_counter()
        query = parse(question, client)
        parsed_ms = (time.perf_counter() - started) * 1000
        context = build(question, query)
        retrieved_ms = (time.perf_counter() - started) * 1000 - parsed_ms

        first_token_ms: list[float] = []
        text = answer(context, client,
                      on_text=lambda _: first_token_ms or
                      first_token_ms.append((time.perf_counter() - started) * 1000))
        total_ms = (time.perf_counter() - started) * 1000

        print(f"\n{'=' * 78}\nQ: {question}")
        print(f"   parse {parsed_ms:.0f} ms | retrieve {retrieved_ms:.0f} ms | "
              f"generate {total_ms - parsed_ms - retrieved_ms:.0f} ms")
        print(f"   FIRST TOKEN {first_token_ms[0]:.0f} ms   (complete at {total_ms:.0f} ms)")
        print(f"   {len(context.recipes)} recipes in context, "
              f"{len(context.numbers())} quotable numbers\n")
        print(text)


if __name__ == "__main__":
    main()
