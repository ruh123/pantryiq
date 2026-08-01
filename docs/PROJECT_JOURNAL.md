# PantryIQ — the project journal

**A plain-language record of what this project is, how it was built, and what we learned at each
step.** No jargon without an explanation. If you read only one document about PantryIQ, read this
one.

> **This is a living document.** A new section is added at the end of every phase. Last updated
> **2026-07-31**, after Phase 3.
>
> Other docs, if you want more depth: `er_metrics.md` (every measurement, including the failed
> ones) · `PantryIQ_Master_Prompt.md` (the locked decisions) · `session_*.md` (day-by-day logs).

---

## 1. What problem is this solving?

Ask a chatbot "how many calories are in this recipe?" and it will give you a confident number.
It usually made that number up. It has no database of foods; it is predicting plausible-sounding
text, and a plausible-sounding calorie count is indistinguishable from a real one.

**PantryIQ's whole design is to make that impossible.** Build a real, verified database of recipe
nutrition first. Then let an AI answer questions — but only from that database, with a
non-AI program checking every number it says.

The hard part is not the AI. It is this:

> A recipe says **"2 cups flour"**. The USDA nutrition database says **"Wheat flour, white,
> all-purpose, unenriched — 364 calories per 100 grams"**. Connecting those two sentences is
> the entire problem.

Two things have to happen, and both can go wrong silently:

1. **Which food is it?** "flour" could be wheat, millet, coconut, almond. Getting this wrong
   means the right *amount* of the wrong *food*. This is called **entity resolution**.
2. **How much is that in grams?** Nutrition is published per 100 grams; recipes are written in
   cups, cans, and "1 onion". Getting this wrong means the wrong *amount* of the right *food*.

Everything below is about doing those two things and — the part that matters most —
**measuring how often we get them wrong.**

---

## 2. The technologies, and why each one

| What | Why it's here, in one sentence |
|---|---|
| **Python 3.11** + **uv** | The language; `uv` installs packages fast and locks exact versions so the project builds identically anywhere. |
| **DuckDB** | A database that lives in a single file — no server to run. Fast for analysis. This is where the cleaned data lives. |
| **Apache Iceberg** | A format for the *raw, untouched* data. It keeps history, so we can always rebuild everything from the original source. |
| **dbt** | Lets us write the data transformations as SQL files with tests attached, instead of scripts nobody can check. |
| **Airflow** | Runs the whole pipeline in the right order, on demand, and can retry a step that fails. |
| **sentence-transformers** | A small AI model that turns text into numbers, so "flour" and "Wheat flour, white" can be compared for *meaning*, not spelling. Runs on this laptop, costs nothing. |
| **Anthropic Claude** | Used sparingly — for labelling data and judging hard cases, never for arithmetic. |
| **pytest** + **ruff** | Tests and a code-style checker. 404 tests currently. |

**A deliberate choice worth understanding:** the matching is done by a *local* AI model, not by
calling a paid API for all 9,163 ingredient names. It's free, it's fast, and it runs offline.

---

## 3. The data

**RecipeNLG** — 2.2 million real recipes scraped from the web. Messy on purpose: we use the
original ingredient lines *with* quantities ("2 cups flour, sifted"), not a cleaned-up version.
A cleaned version would make the matching problem look easier than it is.

> **An early decision that saved the project.** We started with a different dataset (Food.com).
> It stored ingredients as pre-cleaned names with **no quantities** — which would have made the
> matching artificially easy *and* made calorie calculation impossible. We switched to RecipeNLG
> on 2026-07-20 after checking a 500-row sample. Checking the data before building on it is the
> single highest-return habit in this project.

We use a **15,000-recipe subset**, not all 2.2 million. Prove the hard part works small, then
scale.

**USDA FoodData Central** — the US government's food database. We use 8,187 foods (Foundation +
SR Legacy). We deliberately excluded "Branded" foods — millions of supermarket products that
would swamp the matching with near-duplicates.

**What we dropped:** Open Food Facts, a third data source in the original plan, was removed on
2026-07-22 because nothing downstream actually used it. Cutting an unused component is progress.

### How the data is organised — "Bronze, Silver, Gold"

A standard pattern, and the reason for it is simple:

- **Bronze** — the raw data exactly as it arrived, never edited. If we make a mistake later, we
  can always start again from here.
- **Silver** — cleaned and structured. Ingredient lines split into quantity/unit/food; USDA foods
  with their nutrients pulled out.
- **Gold** — the finished, ready-to-use tables. Per-recipe nutrition, cost, dietary tags. This is
  what the AI will read.

Data only ever flows one way: Bronze → Silver → Gold.

---

## 4. Phase 0 — Foundations (2026-07-20)

**Goal:** a project that builds from scratch on a clean machine.

Set up the repository, Python environment, testing, linting, secret handling (API keys live in a
`.env` file that is never committed), and GitHub Actions so tests run automatically on every push.

**Outcome:** `make setup`, `make test`, `make lint` all work. Boring, and the correct place to
start.

---

## 5. Phase 1 — Getting the raw data in (2026-07-22)

**Goal:** land both data sources in Bronze, unmodified, and prove we can read them back.

**What we built:** two ingestion programs — one reads the RecipeNLG file, one calls the USDA API
(paginated, with retries and backoff so a rate-limit doesn't kill the run).

**A design decision that came up immediately:** the plan called for running Iceberg via Docker.
We tried it without Docker first — a local file-based catalog — and it worked. Less infrastructure
to babysit, and the note in the code says to upgrade before deploying.

**Outcome:** 15,000 recipes and 8,187 USDA foods in Bronze, read back successfully.

**A serious bug found by review afterwards:** if the API returned zero rows, the code would have
*replaced the entire table with nothing* — silently, no error. Fixed: a zero-row result is now
refused. This is a recurring theme — **the dangerous failures are the quiet ones.**

---

## 6. Phase 2 — Matching ingredients to foods (the centrepiece)

**Goal:** connect 9,163 distinct ingredient names to USDA foods, and *measure* how often it's right.

**How it works, in order:**

1. **Parse** each line into quantity, unit, and food name. "1 c. firmly packed brown sugar" →
   quantity 1, unit "cup", food "brown sugar".
2. **Deduplicate.** 112,463 ingredient lines collapse to 9,163 distinct names — so every
   expensive step runs 12× less often.
3. **Find candidates.** For each name, retrieve ~78 plausible USDA foods using the local AI model.
   We measured how often the right answer is even *in* that list: **90.5%**. That's a ceiling — you
   cannot pick what was never offered.
4. **Pick one**, by similarity score.
5. **Say "I don't know" when appropriate.** ~12% of ingredient names have no USDA equivalent at
   all ("Stove Top stuffing"). The system declines on those rather than guessing.

### Measuring it — and why this took most of the time

To know if matching is right, you need correct answers to compare against. We hand-labelled
**300 ingredient names**.

> ⚠️ **The most important caveat in the project: 282 of those 300 labels were produced by an AI
> annotator, not a human.** So every accuracy number measures *agreement with those labels*, not
> agreement with truth. We measured the labels' own reliability too — three different AI models
> labelled the same 120 names, agreeing at **α = 0.721**, which is "usable for direction, not for
> precision."

### What we found — including things that didn't work

**The machine learning earned nothing.** We built the "proper" solution: eight engineered
features, a trained model, calibration. It looked better on the data we tuned on and was *worse*
on the held-back test set. When results flip direction like that, it's noise, not skill. **We
shipped the simple version** — just the similarity score — and wrote up why.

**There is a ceiling we cannot pass, and it's in the data.** Even with the *right* food, calories
vary hugely: "pinto bean" is 82 calories/100g canned and 333 dried, and the recipe never says
which. The median spread *within* the correct food is ~32%. A perfect matcher is still that far off.

**We changed what we measure.** Originally: "did we pick the exact right food?" But picking
`Cheese, cheddar` when the label says `Cheese, cheddar, sharp` isn't really an error. So the main
metric became **"is the calorie count within 10%?"** — which is what actually affects a user.

**Where it landed:** 81.0% of ingredient *occurrences* resolve to nutrition within 10% of the
label. (Weighted by how often each name appears, because "salt" appearing 5,000 times matters more
than a name appearing once.)

---

## 7. Phase 2's review — four independent reviews (2026-07-31)

Before moving on, we ran four reviewers over the finished work. **Three of them independently
found the same crash**, which is a strong sign the review was real.

The good news first: the statistics held up. A reviewer re-implemented our agreement statistic
from scratch and matched to four decimal places, and confirmed our confidence intervals use the
right method.

**Seven real defects, all fixed:**

| # | What was wrong | Why it mattered |
|---|---|---|
| 1 | A module crashed on every run | Its own test faked the thing it was testing, so the test suite stayed green while the code was broken |
| 2 | Ties broken by an accident of Python internals | 5% of names tied for best match, and the tie-break decided part of a published result |
| 3 | Five text-parsing bugs | "salt and pepper to taste" became "salt and pepper to" — 479 occurrences confidently matched to *tomato chili sauce* |
| 4 | A control experiment that never actually happened | 40 of 42 "control" rows went to an annotator that never saw the control. **Retracted.** |
| 5 | A comparison that was mathematically guaranteed | Not a measurement at all |
| 6 | An agreement statistic applied inconsistently | 0.709 → 0.721 |
| 7 | A metric labelled "within 10%" that also credited a 5-calorie absolute gap | Relabelled honestly, with a sensitivity table |

**No conclusion reversed.** The machine learning still earned nothing — and read *more* cleanly
afterwards.

**The test suite was the weak point.** We ran 50 deliberate sabotages ("mutations") to see how
many the tests would catch: **14 got through**. Two tests literally described the property they
were checking and then didn't check it. All 14 are now caught.

> **A lesson worth keeping.** The first sabotage run reported "all 14 caught" — and was wrong. The
> test harness was broken, so *everything* failed identically. **A test that fails for the wrong
> reason looks exactly like a test that works.** Always run the control.

---

## 8. Phase 3 — From "which food" to "how many grams" (2026-07-31)

**Goal:** turn "2 cups flour" into an actual weight, then build the finished Gold tables.

**The blocker:** USDA gives nutrition per 100 grams. Recipes say cups and cans. Without a gram
weight, none of the nutrition data is usable.

**What unblocked it:** USDA publishes gram weights ("1 cup, chopped = 160 g") but they weren't in
the data we'd downloaded. We found a different endpoint that serves them — no bulk download
needed. 8,187 foods in ~410 requests.

### Converting to grams — four routes

1. **Already a weight** — "1 lb. beef". Pure arithmetic. Always right.
2. **Package size** — "2 (16 oz.) pkg frozen corn". The parser threw away the "16 oz"; we recover it.
3. **Volume** — "2 cups flour" looked up against that food's published cup weight.
4. **Count** — "1 egg", using a per-item weight.

**When none applies, the answer is empty — never zero.** A zero would claim the ingredient weighs
nothing and quietly drag the recipe's total down while looking like a real measurement.

**Result: 63.9% of ingredient lines convert to a weight.**

### The step that wasn't in the plan, and mattered most

We had measured *how much* we could convert (63.9%) but never *whether the conversions were
right*. Meanwhile Phase 2's accuracy had been measured exhaustively — and every final number is
`grams × calories-per-100g`. We'd verified one factor and never checked the other.

So we built a check: 34 common ingredient+unit pairs compared against **published reference
weights** (1 cup granulated sugar = 200 g, etc. — figures anyone can look up).

**It immediately found something Phase 2 was blind to.** The single most common conversion in the
corpus — "sugar" + "cup", 4,170 lines — read **110 g against a true 200 g**. Bare "sugar" was
matching to *powdered* sugar. And the calorie-based metric could never see it, because powdered
and granulated sugar have nearly identical calories per 100 g. **It only becomes visible when you
weigh it.**

Checking the 40 most common ingredients found more of the same:

| ingredient | occurrences | was matching to | should be |
|---|---|---|---|
| `egg` | 4,399 | **Eggnog** | whole egg |
| `flour` | 2,870 | **Millet flour** | wheat flour |
| `milk` | 2,336 | **Crackers, milk** | milk |
| `cinnamon` | 1,066 | **Bread, cinnamon** | ground cinnamon |

~19,000 occurrences, all matched confidently, none flagged.

**How we fixed it — and the honest caveat.** We first tried a *principled rule* (penalise
derived products like oils, require a whole-word match). Built it, measured it: it fixed `pepper`
and nothing else, while changing 963 names for no measurable gain. **Rejected.**

So we hand-corrected 13 names in a small file, each row recording what it replaced and why.

> **This is an assertion, not a measurement, and the docs say so.** A review showed the apparent
> "improvement" was circular — the only two corrected names the test set could see had been set to
> exactly their test-set answer. The independent gain is **0.0pp**. We kept the corrections anyway,
> because `egg` → *Eggnog* is wrong to any reader whether or not a metric can detect it. But we
> corrected the claim.

### The finished Gold tables

Five tables: the food list, per-ingredient-line details, **per-recipe nutrition**, dietary tags,
and cost.

- **`data_trust_score`** — how much to trust a recipe's numbers, combining *match confidence* with
  *how much of the recipe we could actually weigh*. The original spec used only confidence; that
  would rate a recipe with 40% of its ingredients missing as fully trustworthy.
- **Cost** — a hand-curated table of typical US prices for 98 ingredients. **Clearly labelled a
  stand-in**, not real pricing.
- **Dietary tags** — only for recipes we understand *completely*. An unidentified ingredient is
  exactly the one that might disqualify a "vegan" label.

### The quality gate

Automated checks that **stop bad data reaching Gold**. Twenty checks: every ingredient must point
at a real food, no impossible calorie densities, scores between 0 and 1.

**Thresholds are set to the physically impossible, not the merely unusual** — this corpus really
does contain 40 lb of pork fat and 13 gallons of ice cream. A check that fires on real data gets
switched off, and a switched-off check protects nothing.

**We proved it works** by deliberately poisoning the data and watching Gold refuse to update.

### Phase 3's review found more

Three reviewers again. Four real bugs, all fixed — the worst had real-world consequences:

| bug | before | after |
|---|---|---|
| **False dietary labels** — four rules matched *nothing* | 40 of 429 "gluten-free" recipes contained gluten; 9 of 170 "vegan" contained honey, gelatin or shrimp | **zero** |
| Package sizes multiplied twice | "7 pt. (5 lb.) sugar" → 15,876 g | **2,268 g** |
| Servings misread | "Yield: 2½ dozen" → 2 (should be 30) | **fixed** |
| Per-serving calories mixed partial and complete figures | 94% came from incomplete recipes | only complete ones |

And the gate turned out to be weaker than claimed — bad data *did* briefly land in one Gold table
before being caught. Rather than reword it, we rebuilt it: Gold is now built to a **staging area**,
tested there, and swapped in **all-at-once only if everything passed**. Plus a timestamp on every
build, and a separate small file for serving.

---

## 9. Where the project stands

```
404 tests passing · code style clean · everything committed and pushed
```

| What we measure | Result | What it means |
|---|---|---|
| Ingredient matching | **81.0%** within 10% calories | measured against AI-produced labels |
| Gram conversion accuracy | **77.6%** | against published reference weights |
| Lines converted to a weight | 63.9% | mass 100%, volume 75%, count 38%, packages 27% |
| Recipes with *complete* nutrition | **4.7%** | ← the honest limitation |
| Recipes with complete cost | 0.7% | |

### The finding that matters most

**63.9% of ingredient lines convert, but only 4.7% of recipes are complete.**

Not a contradiction — completeness *compounds*. A recipe needs *every* ingredient converted, so
at ~64% per line, an 8-ingredient recipe has roughly a 1-in-50 chance of being complete.

**This shapes the next phase.** The AI can either answer for that small slice of complete
recipes, or answer for many more *with the coverage stated* ("about 1,400 calories, from the 5 of
8 ingredients we could weigh"). That's a product decision, and it should be made deliberately
rather than discovered halfway through building.

### Known limitations, written down rather than hidden

- **94% of the 300 reference labels were AI-produced.** Every accuracy figure inherits that.
- **The 13 hand-corrections are assertions**, not measured improvements.
- **Cost prices are invented** for realism, not sourced.
- **Some ingredient weights are averaged** where USDA publishes several variants — an average
  is a number USDA never published.
- **A few common ingredients are still mismatched** (`coconut` → coconut oil).

---

## 10. The habit that made the difference

Reading back over three phases, one pattern repeats:

**Every significant error was found by *measuring a claim* rather than re-reading it.**

- The dataset switch — because someone opened the file and looked.
- The sugar mismatch — invisible for a whole phase until we weighed it instead of counting calories.
- The broken test suite — found by deliberately sabotaging the code to see if tests noticed.
- The circular improvement — found by recomputing with the corrections removed.
- The false "all tests caught it" — found by running a control.

And the counter-example, from this project's own history: a claim that the AI matching gains were
"not circular" was written next to real work, sounded reasonable, and was **false**. Nobody
checked it because it read like it had already been checked.

> **The rule this project runs on: a number you haven't recomputed is a number you don't have.**

---

## What gets added here next

Phase 4 (the AI agent) and Phase 5 (the web app and deployment). Each will add a section in the
same shape: *goal → what we built → decisions and why → what we measured → what we got wrong*.
