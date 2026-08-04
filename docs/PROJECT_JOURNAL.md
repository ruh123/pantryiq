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
- The dietary-tag test that agreed with the bug it was testing — found when a new signal
  (recipe titles) finally gave an independent view.
- The "zero problems in 15,000 recipes" that was a broken search — found by asking whether the
  check could fire at all. It couldn't.

And the counter-example, from this project's own history: a claim that the AI matching gains were
"not circular" was written next to real work, sounded reasonable, and was **false**. Nobody
checked it because it read like it had already been checked.

> **The rule this project runs on: a number you haven't recomputed is a number you don't have.**

---

## 11. Phase 4 — The AI agent (2026-08-04)

### The goal

Put an AI assistant on top of everything built so far — and make it so it **can only tell you
things the warehouse actually knows**. Not "we asked it nicely to be accurate". Actually checked,
by code, on every single answer.

### How it works — four steps, and only two involve AI

```
  you ask a question
        |
   1.  PARSE      Claude turns your words into a search filter
        |         (it may shape the search — but it can only fill in a fixed form)
   2.  RETRIEVE   plain database query against the verified data
        |         (no AI at all — this is the important part)
   3.  GENERATE   Claude writes the answer, and is shown ONLY what step 2 found
        |         (it has no access to the recipe database, no tools, nothing else)
   4.  GUARDRAIL  ordinary code checks every number in the answer
                  against what step 2 returned. Anything else is rejected.
```

The reason step 2 has no AI in it is the whole design. If the model could choose what to look up,
"it only speaks from verified data" would be a hope. Because retrieval happens in code, it's a
fact about how the program is built.

### The guardrail — the part we're proudest of

After Claude writes an answer, code pulls out **every number in it** and checks each one against
the data. Not "does this look right" — literally, is this number in the retrieved facts?

We then tested it the only honest way: we took 20 real answers and deliberately corrupted them —
changed a calorie count, invented a price, made up a serving count, divided a total by servings,
added a plausible-sounding round number. Then we counted how many the guardrail caught.

| | result |
|---|---|
| **fabricated numbers caught** | **36 out of 36 (100%)** |
| **correct answers wrongly blocked** | **0 out of 20 (0%)** |
| numbers checked in total | 346 (about 17 per answer) |

Both numbers matter. A guardrail that rejects *everything* would catch 100% of lies and be
useless. Reporting only the catch rate would hide that.

### The number the AI is not allowed to compute

This one is subtle and it's the best example of the whole approach.

A recipe might say "makes 8 servings" and "total: 3,470 calories". Any AI will happily divide and
tell you "434 calories per serving". But back in Phase 3 we deliberately **refused** to publish
that division, because that recipe's calorie total was missing an ingredient we couldn't weigh —
so the total is an undercount, and dividing an undercount by a real serving count gives you a
confident-looking number that's wrong.

So the guardrail does not allow division. 434 isn't in the data, so the answer is rejected. When
we ran it, Claude wrote *"It makes 8 servings, though the per-serving calories aren't available"*
— which is exactly right.

### Something broke, and it was our fault from Phase 3

Once the agent could name recipes, we could finally cross-check the dietary tags — and found a
recipe called **"Meat Loaf" tagged vegetarian**.

Here's why. Phase 3 decided "vegetarian" by checking that no ingredient's name contained a meat
word (beef, pork, chicken...). But this recipe's "hamburger" had been matched to a USDA food
called **"BURGER KING, Hamburger"** — which contains none of those words. So it passed.

Worse: **our own test for this problem used the same method**, so it reported everything was
fine. The test and the thing it was testing shared a blind spot.

**The fix was to turn the logic inside out.** Instead of a list of *banned words* (where anything
you didn't think of slips through), we downloaded USDA's own food categories — there are exactly
25, things like "Beef Products", "Vegetables and Vegetable Products", "Fast Foods" — and now a
recipe is only tagged vegetarian if **every** ingredient is in a category we've allowed. Anything
unrecognised fails. "BURGER KING, Hamburger" is in "Fast Foods", which isn't on the list, so it
declines instead of lying.

Result: recipes falsely tagged vegetarian went from 13 to 5, vegan from 9 to 4, gluten-free from
329 tagged down to 63 (a much smaller but much more trustworthy set).

### The discovery underneath that one

Chasing the last few leaks turned up something more interesting: **the original recipe data is
sometimes incomplete.**

A recipe called "Pickled Bologna" lists exactly four ingredients: vinegar, sugar, salt, pickling
spice. **There is no bologna in the ingredient list.** Our pipeline handled it perfectly — it
faithfully recorded all four ingredients. But that means the recipe passes every check we have
and is not vegetarian.

The lesson: "we weighed 100% of the ingredients" is not the same as "we have all the
ingredients", and nothing downstream of the ingredient list can tell the difference.

The cooking instructions *can* — they say "remove skin from 2 rings bologna". So we now also
check the method text, and refuse to tag a recipe when the instructions mention a food the
ingredient list doesn't.

**And our first attempt at measuring this was silently broken.** It reported zero problems across
all 15,000 recipes, which looked like great news. In fact the instructions field was being read
incorrectly (it's stored as text, and our code treated it as a list, turning "boil ingredients"
into "b o i l   i n g r e d i e n t s"), so the search could never match anything. We only caught
it by asking "would this test ever fire?" — and the answer was no. Fixed, it fires on 20.6% of
recipes.

### Speed — we missed the target and are saying so

The goal was under 5 seconds from question to answer. Measured:

| step | time |
|---|---|
| Claude understands your question | 3.6 – 4.0 s |
| **database search** | **0.015 – 0.038 s** |
| Claude starts writing | ~1.3 – 2.3 s more |
| **first words appear** | **5.3 s** (median) |
| whole answer finished | 7 – 12 s |

The database part is essentially free. All the time is the two AI calls, and both were already
set to their fastest settings. We added **streaming** so words appear as they're written, which
took the wait from 13.4 seconds to 5.3 — a big improvement, still just over target.

A faster model would close the gap (we measured one that does the first step in 2.4s instead of
3.6s with identical results), but swapping models is your call, not a change to make quietly.

One thing we did fix: the database search was originally taking **973 milliseconds** because of
how the search query was written. Rewritten, it takes **17** — 57× faster, same results.

### The meal planner — where the AI does the least

"Plan a week under $50 with no day over 2,000 calories."

The **plan is chosen by ordinary code**, then verified by ordinary code, and only then does
Claude describe it. The AI does no arithmetic and no constraint-solving — the brief was explicit
about that, and it's right: arithmetic is the thing computers already do perfectly.

A real run: 7 days, $33.65 of the $50 budget, every day under the ceiling, verified in code.

This exposed one more thing worth recording. Because we'd told Claude never to add numbers up, it
**refused to describe its own plan** — "I can't total up costs across days". Correct behaviour,
wrong outcome. So we added a separate channel for figures that were *already calculated and
checked by code*, which the AI may quote. Those are results, not arithmetic it performed.

### The quality explainer — the same idea pointed at the engineers

The last piece serves whoever maintains this rather than whoever cooks from it. When a data
quality check fails, or when the matcher wasn't confident about an ingredient, the agent writes a
plain-English note into a runbook table: what broke, what it means for the published data, and
what to look at first.

We made it prove itself the same way we proved the quality gate. We deliberately broke the
pipeline — injected an impossible food (a calorie value no food can have) — captured exactly what
the failure looked like, and pointed the explainer at it. It wrote:

> "An accepted_range test on `canonical_ingredients.kcal_per_100g` failed: 1 row falls outside the
> configured bounds of 0 to 910, and because the test failed, 17 downstream models were skipped
> and did not build. That means anything depending on canonical ingredients is stale — consumers
> are reading the previous run's output, not fresh data... start by querying for `kcal_per_100g`
> below 0 or above 910 to identify the offending ingredient."

That's a genuinely useful thing to be woken up by. **And the first time we ran it, our own
guardrail rejected it** — for two reasons, both our fault:

- It quoted the column name `kcal_per_100g`, and our number-checker read the "100" in that name as
  a made-up figure. It's part of a *name*, not a measurement. (Exactly the same mistake we'd
  already made once with recipe IDs.)
- It quoted the allowed range "0 to 910" — which is real, but we hadn't put those bounds into the
  list of facts it was allowed to quote. So a true statement was rejected.

Both fixed, and the same failure now produces a summary that passes cleanly. This is the pattern
the whole project runs on: the check fired, we looked at *why*, and the answer was that the check
was wrong rather than the output.

The explainer is held to the same standard as answers to users — it can't invent a number either.
An ops summary with a made-up row count is worse than none at all, because someone will act on it
without the query in front of them. If the drafted summary fails the check, the runbook gets the
raw facts instead.

### Then we reviewed it all, and found we'd published two wrong numbers

Five independent reviewers went over the finished phase. They found that **both of the headline
claims were false** — and in both cases the mistake was structural, not arithmetic.

**1. "The guardrail catches 100% of fabrications" could not have said anything else.**

The way we tested it: corrupt an answer, see if the guardrail catches it. But we first *skipped*
any corruption the guardrail would allow, on the grounds that it "wasn't really a fabrication" —
using the same piece of code that then decided whether it was caught. So the test agreed with
itself 90 times out of 90. A guardrail that catches nothing would have scored 100% too.

Counting every corruption honestly, the real figure was **40%**. The reason: our "close enough"
tolerance was ±10%, borrowed from a different part of the project where it means "these two foods
are nutritionally interchangeable". Applied to quoting accuracy, it accepted **79% of all possible
numbers**. And it wasn't buying anything: of 346 numbers in 20 real answers, 345 were exact
matches and one was a bullet point. Not one needed the tolerance.

**Rebuilt**, with three changes:
- Numbers are now checked against **the recipe the sentence is about**, not the whole answer. The
  old version accepted "Favorite Chicken is 1,968 calories" when 1,968 belonged to a different
  dish — an error we measured at typically 133% and up to 1,105% wrong.
- Counts and measurements are separated properly. The old rule rejected "about 13 g of protein"
  against a stored 13.1, on **half** of all such sentences.
- Its own tolerance (1%), not a borrowed one.

Result: **86.5% caught** (up from 40%), misattribution 100% caught, and no genuine false alarms.
The one answer it flagged turned out to be a **real** model error — Claude wrote *"Cost is at
least $2.22 — sorry, at least $3.65"*, inventing a number and correcting itself, and the guardrail
caught the invented one. That's the first time it has ever caught a genuine mistake in the wild.

**2. The dietary tags were still wrong, six times more often than we said.**

We'd reported 13 falsely-tagged vegetarian recipes. Checking the actual ingredients rather than
the recipe names, it was **30 out of 572** — and 16 of 134 vegan. Two recipes tagged *vegan* were
a ribs recipe and a cocktail-wieners recipe.

Why: Worcestershire sauce contains anchovies. Marshmallows contain gelatin. Neither has a meat
word in its name, and both sit in food groups we'd allowed. We'd fixed the rule twice and both
times fixed a *proxy* for the real question rather than the question itself.

**So we asked the question directly.** Every one of the 8,187 USDA foods is now individually
classified — is this vegetarian? vegan? gluten-free? — and a recipe only gets a tag if *every*
ingredient qualifies. Anything uncertain counts as "no". A food nobody has ever looked at gets
classified the same way as one that has, which is the property the previous two rules lacked.

Worth recording: our hand-written list of "correct answers" for testing the classifier was itself
wrong in four places — all in the unsafe direction. It claimed Worcestershire sauce is
gluten-free (it contains barley malt vinegar) and russian dressing is vegetarian (it usually
contains Worcestershire, hence anchovy). **The classifier corrected us.**

**And the safety checks now actually block.** They used to live only in the Python tests, which
run on a developer's machine — meaning a false "vegan" label could be built and published with
nothing stopping it. They're now part of the build gate. We proved it by deliberately breaking
the rule: the build failed on 371 bad rows and refused to publish.

### Two more checks that couldn't fire

The review also caught a pattern we've now hit three times: a check written, never watched, and
silently incapable of firing.

- A gluten check used a SQL operator where `%` means a literal percent sign rather than "anything"
  — so it matched nothing, ever.
- The mutation test that measures whether our tests are any good was itself broken: it forgot to
  copy one directory, so every run failed for the same unrelated reason and reported success.

Running the fixed version: **62 deliberate bugs introduced, 32 slipped past our tests.** The worst
were two functions with *no tests at all* — including the one that decides whether a rejected
answer gets shown to the user. That one was genuinely broken: it recorded a clean answer in the
log alongside a "failed" verdict and numbers that appeared nowhere in it.

Test count is now **556**, up from 520.

### What we can now do

```
uv run python -m pantryiq.agent "what can I make with chicken, rice and onions?"
uv run python -m pantryiq.agent.planner
uv run python scripts/measure_guardrail.py
uv run python -m pantryiq.agent.explain          # runbook entries for failures
uv run python -m pantryiq.er.dietary --check     # the dietary classifier's known answers
```

Every question is logged — the question, what was retrieved, the answer, and the guardrail's
verdict — so any past answer can be re-checked later.

### Honest limitations

- **The guardrail checks that a number exists in the data, not that it belongs to the recipe
  being discussed.** Quoting one recipe's calories while naming another would pass.
- **Speed misses the 5-second target** (5.3s to first words).
- **Some dietary tags are still wrong** — about 5 in 572 vegetarian recipes, mostly where the
  source recipe data itself is incomplete.
- **The planner's picks skew to desserts**, because only 42 recipes have both complete nutrition
  and complete pricing, and cakes are calorie-dense.
- **Recipe relevance has no concept of "dinner"** — ask for a vegetarian dinner and you may get
  cookies, because the data has no meal-type information. The agent says so rather than pretending.

---

## What gets added here next

Phase 5 (the web app and deployment). It will add a section in the same shape:
*goal → what we built → decisions and why → what we measured → what we got wrong*.
