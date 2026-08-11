"""The web app — the architecture made visible, with a chat box attached.

The brief is blunt about this (§2): "the chat UI is the least differentiating part." Any recipe
app can print an answer. What this one can do is show, beside every answer, the exact set of facts
the model was permitted to speak from and a deterministic verdict on every number it wrote. So
the ledger and the badge are the screen, and the prose sits above them.

Three deliberate choices:

**Nothing streams.** `guardrail.py` needs a complete response before any of it can be trusted, so
showing tokens as they arrive would mean showing text that may be retracted. The wait is spent on
stage labels instead - which are also the honest thing to show, because the stage breakdown is
itself the argument: retrieval decides what may be said and costs tens of milliseconds, while the
two model calls cost seconds.

**Numbers are rendered straight off `RecipeFact` and never re-rounded.** `context.py` rounds once,
before the prompt and the guardrail both see the values, so the screen agrees with the check by
construction. Formatting a figure differently here would put a number on the page that the
guardrail never approved.

**The caveats are structural, not editorial.** A fluent answer can hide a truncated ingredient
list or a stand-in price. They are shown from `published.py` whether or not the prose mentions
them.

Run:  uv run streamlit run src/pantryiq/serving/app.py
"""
from __future__ import annotations

from html import escape

import duckdb
import streamlit as st

from pantryiq.agent.claude import get_client
from pantryiq.agent.retrieval import DEFAULT_DB as GOLD_DB
from pantryiq.serving import published
from pantryiq.serving.answer import (
    Answered,
    ServingError,
    answer_question,
    plan_week,
    startup_problem,
)
from pantryiq.serving.recipes import directions_for
from pantryiq.serving.style import CSS, meter, stat_tile

# A public URL in front of a metered API key. Neither is a security boundary — a determined
# visitor can clear session state — but together they bound the cost of ordinary traffic.
MAX_QUESTION_CHARS = 300
MAX_QUESTIONS_PER_SESSION = 20

# (chip, question). The chips are short so the row keeps one height — the full questions wrapped
# to two lines and left the row ragged.
EXAMPLES = (
    ("Chicken, rice, onions", "What can I make with chicken, rice and onions?"),
    ("Vegetarian under 500 kcal", "I want a vegetarian dinner under 500 calories"),
    ("Something we don't stock", "What can I cook with unobtainium and moon cheese?"),
)

STAGE_LABELS = {
    "parsing": "Reading the question into a filter (Claude)",
    "retrieving": "Searching the warehouse (SQL, no model)",
    "answering": "Writing the answer, then checking every number in it",
    "selecting": "Choosing the week in code",
    "narrating": "Describing the plan, then checking every number in it",
}

st.set_page_config(page_title="PantryIQ", page_icon="🥫", layout="wide")


@st.cache_resource
def claude():
    """One client per server process. Built here so its cost is not charged to a question."""
    return get_client()


@st.cache_data
def provenance() -> dict | None:
    """The stamp `gold/publish.py` writes into the artifact — which build is being served."""
    try:
        con = duckdb.connect(str(GOLD_DB), read_only=True)
    except Exception:  # noqa: BLE001 - provenance is a nicety; its absence must not blank the app
        return None
    try:
        row = con.execute(
            "SELECT published_at, git_sha, gold_row_counts FROM gold.pipeline_run LIMIT 1"
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    finally:
        con.close()
    if row is None:
        return None
    return {"published_at": row[0], "git_sha": row[1], "row_counts": row[2]}


def table(headers: tuple[str, ...], rows) -> None:
    """A markdown table.

    Not `st.table`/`st.dataframe`: those route a list of dicts through pandas, which hands the
    strings to pyarrow, which **segfaulted** on this pandas 3.0.5 / pyarrow 25.0.0 pair inside
    Streamlit's script thread — a hard crash, not an exception, so no error boundary would have
    caught it in production. Nothing on this page is larger than eight rows or interactive, so the
    dataframe stack was buying nothing and risking the whole page.
    """
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    st.markdown("\n".join(["| " + " | ".join(headers) + " |", rule, body]))


def md(text: str) -> str:
    """Escape `$` before any model or warehouse text reaches a markdown renderer.

    Streamlit reads `$…$` as inline LaTeX, so two dollar amounts in one paragraph swallow
    everything between them into a serif formula: "comes to $33.65 total, leaving $16.35 of your
    $50 budget" rendered as maths. Costs are the one figure this app quotes most, and the pairing
    is invisible until a paragraph happens to contain two of them.

    Display only — the guardrail checks the unescaped text, so what is verified and what is shown
    stay the same numbers.
    """
    return text.replace("$", "\\$")


def money(value: float | None, coverage: float | None) -> str:
    """A partial cost is a floor. Saying "$5.43" for three of five priced ingredients is a claim
    about the dish nobody measured — `generate.py`'s system prompt rule 7 in the UI's own voice."""
    if value is None:
        return "not priced"
    if coverage is not None and coverage < 1.0:
        return f"at least ${value:,.2f}"
    return f"${value:,.2f}"


def render_verdict(result: Answered) -> None:
    """The badge. Four states, because `regenerated` alone conflates two of them."""
    checked = result.verdict.checked
    if not result.verdict.passed:
        st.error(
            f"**Guardrail FAIL** — of {checked} numeric claims, these trace back to nothing in "
            f"the retrieved data: {', '.join(f'{n:,g}' for n in result.verdict.unsupported)}"
        )
    elif result.fell_back:
        st.warning(
            f"**Guardrail rejected two drafts.** Rather than show an unverifiable answer, this "
            f"is the warehouse data stated directly. {checked} numeric claims checked."
        )
    elif result.regenerated:
        st.warning(
            f"**Guardrail rejected the first draft**, and the regenerated one passed on "
            f"{checked} numeric claims. Every figure above traces to the retrieved data."
        )
    else:
        st.success(
            f"**Guardrail PASS** — {checked} numeric claims checked, every one traced back to "
            f"the data below. Nothing here came from the model's own knowledge."
        )
    if result.log_error:
        st.caption(f"Not written to the query log: {result.log_error}")


def render_timings(result: Answered) -> None:
    spans = "  ·  ".join(f"{STAGE_LABELS.get(name, name).split(' (')[0].lower()} "
                         f"**{ms:,.0f} ms**" for name, ms in result.timings)
    st.caption(f"{spans}  ·  total **{result.latency_ms:,.0f} ms**")


def render_ledger(result: Answered) -> None:
    """Everything the answer was allowed to say. Values come off `RecipeFact` unchanged."""
    recipes = result.context.recipes
    if not recipes:
        st.info("Retrieval returned nothing, so the answer had no facts to draw on at all.")
        return

    st.markdown(f"#### What the answer was allowed to say — {len(recipes)} recipes")
    # The ranking sentence has to match the query that produced it: with a dish named, every row
    # matched the title and there may be no pantry terms at all, so claiming a pantry ranking
    # would describe a rule that did not run.
    ranking = ("Ranked by how many of your pantry items each uses, ties broken on trust score."
               if result.context.query.pantry else
               "Every one matches the dish you named; ranked by how completely we could verify "
               "the recipe.")
    st.caption(
        f"{ranking} The model saw exactly this and nothing else — the cooking methods below are "
        "read separately, for you, and are never shown to it."
    )

    # One query for every card. The method is fetched here rather than inside the loop because
    # eight connections would cost more than the retrieval that produced these recipes.
    methods = directions_for([recipe.recipe_id for recipe in recipes])

    for recipe in recipes:
        kcal = ("not weighed" if recipe.total_kcal is None
                else f"{recipe.total_kcal:,} kcal for the whole dish")
        weighed = (f"{recipe.counted_ingredients} of {recipe.ingredient_count} "
                   "ingredients weighed")
        if recipe.counted_ingredients != recipe.ingredient_count:
            weighed += ", so this is an undercount"
        per_serving = ("no per-serving figure is published for this recipe"
                       if recipe.kcal_per_serving is None
                       else f"{recipe.kcal_per_serving:,} kcal per serving")
        sub = []
        if recipe.matched:
            sub.append(f"uses {', '.join(recipe.matched)}")
        if recipe.missing:
            sub.append(f"missing {', '.join(recipe.missing)}")
        tags = "".join(f'<span class="rc-tag">{tag}</span>' for tag in recipe.tags)

        left, right = st.columns([3, 2], gap="medium")
        with left:
            st.markdown(
                f'<div class="rc-title">{recipe.title or recipe.recipe_id}</div>'
                f'<div class="rc-sub">{" · ".join(sub) or "&nbsp;"}</div>'
                f'<div class="rc-fact"><b>{kcal}</b> — {weighed}</div>'
                f'<div class="rc-fact">{money(recipe.cost_total_usd, recipe.cost_coverage)}'
                f' · {per_serving}</div>'
                f'<div style="margin-top:.35rem">{tags}</div>',
                unsafe_allow_html=True)
        with right:
            st.markdown(
                meter("data_trust_score", recipe.data_trust_score)
                + meter("nutrition coverage", recipe.nutrition_coverage, severity=True),
                unsafe_allow_html=True)

        steps = methods.get(recipe.recipe_id, ())
        if steps:
            with st.expander(f"Method — {len(steps)} steps"):
                st.markdown(
                    "".join(f'<div class="step"><span class="step-n">{index}</span>'
                            f"<span>{escape(step)}</span></div>"
                            for index, step in enumerate(steps, 1)),
                    unsafe_allow_html=True)
                st.caption(
                    "Scraped text, shown as published. It is not part of what the answer above "
                    "was checked against — compare it with the ingredient list, because this "
                    "corpus ships recipes whose ingredients are incomplete."
                )
        else:
            st.caption("No method published for this recipe.")
        st.divider()


def run_question(question: str) -> None:
    """One question, start to finish, with the stage shown while it runs."""
    with st.status("Working…", expanded=True) as status:
        def on_stage(name: str) -> None:
            status.update(label=STAGE_LABELS.get(name, name))

        try:
            result = answer_question(question, client=claude(), on_stage=on_stage)
        except ServingError as exc:
            status.update(label="Could not answer that", state="error")
            st.error(str(exc))
            return
        status.update(label=f"Answered in {result.latency_ms / 1000:,.1f} s", state="complete")

    st.markdown(f'<div class="answer">\n\n{md(result.text)}\n\n</div>', unsafe_allow_html=True)
    render_verdict(result)
    render_timings(result)
    render_ledger(result)


def stage_strip() -> None:
    """The four stages, shown while the page is otherwise idle.

    It fills the space under the question box, and it is the argument the product is making:
    retrieval decides what may be said and no model touches it. Two of the four are code.
    """
    cards = []
    for index, (name, who, what) in enumerate(published.PIPELINE, 1):
        css = "who-model" if who == "Claude" else "who-code"
        cards.append(
            f'<div class="stage"><div class="stage-n">STEP {index}</div>'
            f'<div class="stage-name">{name}</div>'
            f'<span class="stage-who {css}">{who}</span>'
            f'<div class="stage-what">{what}</div></div>')
    st.markdown("#### How an answer gets made")
    st.markdown('<div class="stages">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="stamp" style="margin-top:.9rem">Only steps 1 and 3 involve a model, and '
        f"neither chooses the facts. Every number in the finished answer is traced back to what "
        f"step 2 returned — {published.GUARDRAIL_CAUGHT} deliberately injected fabrications were "
        f"caught this way ({published.GUARDRAIL_CATCH_RATE}).</div>", unsafe_allow_html=True)


def ask_tab() -> None:
    asked = st.session_state.get("asked", 0)

    with st.form("ask", clear_on_submit=False, border=False):
        box, send = st.columns([6, 1], vertical_alignment="bottom")
        question = box.text_input(
            "Your question", placeholder=EXAMPLES[0][1], max_chars=MAX_QUESTION_CHARS,
            label_visibility="collapsed", value=st.session_state.get("question", ""),
        )
        submitted = send.form_submit_button("Ask", type="primary", use_container_width=True)

    columns = st.columns(len(EXAMPLES) + 1)
    for column, (chip, full) in zip(columns, EXAMPLES, strict=False):
        if column.button(chip, use_container_width=True):
            st.session_state["question"] = full
            st.rerun()

    if not submitted or not question.strip():
        stage_strip()
        return
    if asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            f"This demo allows {MAX_QUESTIONS_PER_SESSION} questions per session — the API key "
            "behind it is metered. Reload the page to start a new one."
        )
        return

    st.session_state["asked"] = asked + 1
    run_question(question.strip())


def plan_tab() -> None:
    st.markdown("#### Plan a week, solved in code")
    st.caption(
        "The selection and the arithmetic happen in Python and are verified before Claude sees "
        "them; the model only describes a decision already made. It plans on recipe **totals** — "
        f"only {published.COMPLETE_COST_N} recipes have both complete nutrition and complete "
        "cost, and how many people a dish feeds is the cook's judgement, not ours."
    )

    left, middle, right = st.columns(3)
    budget = left.number_input("Budget (USD)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)
    kcal = middle.number_input("Calories per day", min_value=500.0, max_value=5000.0,
                               value=2000.0, step=100.0)
    days = right.number_input("Days", min_value=1, max_value=14, value=7, step=1)

    if not st.button("Plan it", type="primary"):
        return

    with st.status("Working…", expanded=True) as status:
        def on_stage(name: str) -> None:
            status.update(label=STAGE_LABELS.get(name, name))

        try:
            week = plan_week(float(budget), float(kcal), int(days), client=claude(),
                             on_stage=on_stage)
        except ServingError as exc:
            status.update(label="Could not plan that", state="error")
            st.error(str(exc))
            return
        status.update(label=f"Planned in {week.narration.latency_ms / 1000:,.1f} s",
                      state="complete")

    plan = week.plan
    if week.violations:
        st.error(md("The plan breaks its own constraints: " + "; ".join(week.violations)))
    elif len(plan.recipes) < plan.days:
        st.warning(
            f"Only {len(plan.recipes)} of {plan.days} days could be filled without repeating a "
            "recipe or exceeding the budget. A short week is an honest answer."
        )
    else:
        st.success(md(
            f"Constraints verified in code before anything was written: "
            f"${plan.total_cost:,.2f} of ${plan.budget_usd:,.2f}, no day above "
            f"{plan.kcal_per_day:,.0f} kcal."
        ))

    table(("day", "recipe", "kcal", "cost"),
          [(index, recipe.title or recipe.recipe_id, f"{recipe.total_kcal:,.0f}",
            md(f"${recipe.cost_total_usd:,.2f}"))
           for index, recipe in enumerate(plan.recipes, 1)])

    st.markdown(md(week.narration.text))
    render_verdict(week.narration)
    render_timings(week.narration)


def how_tab() -> None:
    st.markdown("#### The headline is entity resolution, not the chat")
    st.write(
        f"{published.INGREDIENT_LINES} free-text ingredient lines from {published.RECIPES} "
        f"recipes, reduced to {published.DISTINCT_STRINGS} distinct strings and resolved against "
        f"{published.USDA_ENTITIES} canonical USDA food entities. \"2 cups flour, sifted\" has to "
        "become a specific USDA food before anything downstream can be true. That is the hard "
        "part, and it is measured rather than asserted."
    )
    table(("measured", "per unique string", "per occurrence"), published.ENTITY_RESOLUTION)
    st.caption(
        f"The resolver commits to an entity for {published.RESOLVER_COMMITS} of strings and "
        f"declines on {published.RESOLVER_DECLINES} rather than guessing, because a wrong match "
        f"silently produces wrong nutrition where an honest gap does not. Accuracy is conditional "
        f"on not declining, so neither column is quotable alone. {published.LLM_LABELS} gold "
        f"labels were AI-produced: these measure agreement with those labels, not with truth."
    )

    st.divider()
    st.markdown("#### Four stages, and only two involve a model")
    table(("stage", "who decides", "what it does"), published.PIPELINE)
    st.caption(
        "Retrieval sitting outside the model's control is what makes \"it can only speak from "
        "verified data\" checkable rather than aspirational. The guardrail then extracts every "
        "number from the response and requires each to trace back to a retrieved value — in "
        "code, not by asking another model."
    )

    st.divider()
    st.markdown("#### The guardrail, measured against deliberate sabotage")
    table(("what was measured", "result"), published.GUARDRAIL)
    st.caption(
        "Nine classes of fabrication were injected into real answers and the guardrail was scored "
        "on how many it rejected. An earlier version of this table claimed 100%; that figure was "
        "a tautology — the harness used the same predicate to plant an error and to judge it "
        "caught. On an honest denominator the old rule scored 40%."
    )

    st.divider()
    st.markdown("#### What the warehouse can and cannot answer")
    table(("what was measured", "result"), published.COVERAGE)
    st.caption(
        "Coverage compounds: a recipe needs every line weighed, so a per-line rate of "
        f"{published.LINES_WEIGHED} leaves only {published.COMPLETE_NUTRITION} of recipes "
        "complete. Dietary tags: "
        + ", ".join(f"{name} {count}" for name, count in published.TAGS) + "."
    )

    st.divider()
    st.markdown("#### Read these before quoting anything")
    for title, body in published.CAVEATS:
        with st.expander(title):
            st.write(body)


def sidebar() -> None:
    """Provenance and scope. The caveats used to live here and dominated the first screen —
    eight identical grey accordions, making the most defensive part of the product the loudest
    thing on it. They belong with the rest of the measurement story, in the third tab."""
    with st.sidebar:
        st.markdown("### PantryIQ")
        st.markdown(
            '<div class="stamp">A verified recipe data platform with a thin AI layer that can '
            "only speak from it.</div>", unsafe_allow_html=True)
        stamp = provenance()
        if stamp:
            st.markdown(
                f'<div class="stamp" style="margin-top:.8rem">Serving the Gold build published '
                f"<b>{stamp['published_at']:%Y-%m-%d %H:%M}</b><br>from commit "
                f"<code>{stamp['git_sha']}</code></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown(
            f'<div class="stamp"><b>Scope.</b> A {published.RECIPES}-recipe subset of RecipeNLG, '
            f"resolved against {published.USDA_ENTITIES} USDA foods. Prices are a curated "
            "stand-in, not live pricing. Every limitation is listed under "
            "<b>How this works</b>.</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="stamp" style="margin-top:.8rem">No accounts, no history, nothing stored '
            "about you. Questions and the facts retrieved for them are logged so any past answer "
            "stays checkable.</div>", unsafe_allow_html=True)


def kpi_row() -> None:
    """The four figures the project is judged on, on screen before anything is asked.

    §2 of the brief: lead with the entity-resolution problem and the metrics, not the chat UI.
    The landing view previously opened on an empty question box and said nothing about what the
    project had actually measured. A KPI row is the right form for four standalone numbers —
    they share no scale, so a chart comparing them would invite a comparison that means nothing.

    The fourth tile is the limitation, not an achievement. It is here on purpose.
    """
    st.markdown(
        '<div class="kpi-row">'
        + stat_tile("Ingredient lines resolved", published.INGREDIENT_LINES,
                    f"free text to {published.USDA_ENTITIES} canonical USDA entities")
        + stat_tile("Nutrition match, per occurrence", "81.0%",
                    "within 10% of the gold label · conditional on not declining")
        + stat_tile("Fabrications caught", published.GUARDRAIL_CATCH_RATE,
                    f"{published.GUARDRAIL_CAUGHT} injected errors · 0 genuine false positives")
        + stat_tile("Recipes fully weighed", published.COMPLETE_NUTRITION,
                    f"{published.COMPLETE_NUTRITION_N} of {published.RECIPES} — coverage "
                    "compounds, and this is the honest ceiling")
        + "</div>", unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    problem = startup_problem()
    if problem:
        st.title("PantryIQ")
        st.error(problem)
        st.stop()

    sidebar()
    st.title("PantryIQ")
    st.markdown(
        '<div class="stamp" style="font-size:.85rem">Free-text recipe ingredients resolved to '
        "canonical USDA foods, with a deterministic check on every number the AI writes.</div>",
        unsafe_allow_html=True)
    kpi_row()
    ask, plan, how = st.tabs(["Ask", "Plan a week", "How this works"])
    with ask:
        ask_tab()
    with plan:
        plan_tab()
    with how:
        how_tab()


# Streamlit executes this file as `__main__`, so the guard is a no-op in production — but without
# it, merely importing the module to reach a helper runs the whole app against whatever Streamlit
# context happens to be current. In a test process that means a half-built page and a form left
# open, which then breaks the next real run. Found exactly that way.
if __name__ == "__main__":
    main()
