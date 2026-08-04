# Entity resolution — measured results

Regenerate: `uv run python -m pantryiq.er.metrics` (recall@k) ·
`uv run python -m pantryiq.er.nutrition` (kcal error + ambiguity ceiling) ·
`uv run python -m pantryiq.er.convention` (rule 2b audit) ·
`uv run python -m pantryiq.er.report` (2.5 holdout) ·
`uv run python -m pantryiq.er.resolve` (shipped resolver) ·
`uv run python -m pantryiq.er.relabel --report --pass 1` ·
`uv run python -m pantryiq.er.adjudicate --report` ·
`uv run python -m pantryiq.er.ensemble --report` (annotator agreement) ·
`uv run python -m pantryiq.er.adjudicator_value --report` (2.6 viability) ·
`uv run python -m pantryiq.er.entity_map` (2.7 entity map + corpus headline) ·
`uv run python -m pantryiq.er.abstain` (no-match threshold)

**If you read one thing:** the headline metric of this project is **nutrition error, not entity
accuracy**, and the reason is measured rather than asserted — see
[The ambiguity ceiling](#the-ambiguity-ceiling). A median ~32% kcal spread exists *within* the
correct food, so no entity-level precision figure can carry the weight the plan originally
assigned it.

> **All figures below were regenerated on 2026-07-31** after an external review found a
> tie-break bug and five parser artifacts. See [§13](#13-the-review-pass-2026-07-31) for what
> changed and why, including the old→new table. Earlier revisions of this file quote the
> pre-fix numbers.

**The 2.5 result, on the frozen holdout: 68.4% of resolved strings are nutritionally equivalent
to the gold label (95% CI 58.2–78.5%), median kcal error 0.0%.** Eight engineered features and a
calibrated logistic model are worth nothing measurable over taking the top embedding cosine — see
[§7](#7-scoring-and-routing-25--the-frozen-holdout-read-once).

**The corpus headline (2.7): 77.4% of ingredient *occurrences* resolve to nutrition within 10%
— or within 5 kcal/100g — of the gold label** (per unique string: 69.2%), with the resolver
declining on 12.5% of occurrences rather than guessing (§12) — a stratified estimate over all
9,163 strings, see [§11](#11-the-entity-map-and-the-corpus-headline-27).

**Label quality, measured: Krippendorff's α = 0.721 [0.651, 0.787]** across three independent
annotators on 120 strings
([§9](#9-multi-annotator-ensemble--α--0709-and-it-does-not-beat-the-gold-labels)). The interval
straddles the ~0.67 floor, so quote it rather than the point estimate — usable for directional
claims, not precise ones. The ensemble could not improve on the existing labels, so they were
left in place.

---

## 1. Label provenance — read before quoting any number

The 300 gold labels are **not all human-produced**:

| Labeler | Count | What it means |
|---|---|---|
| human | 18 | Independent ground truth |
| **claude (LLM annotator)** | **282** | Model-produced, following `docs/labeling_guide.md` |

A deliberate trade: hand-labeling 300 strings costs 8–12 hours, and the project owner chose LLM
annotation over abandoning the metric. Consequences that must travel with every number below:

- **Entity-level figures measure agreement between the pipeline and an LLM annotator**, not
  accuracy against human ground truth.
- **A Claude adjudicator (2.6) cannot be honestly scored against these labels** — that is
  self-consistency. Score it on human-labeled rows only, or do not claim it.
- **recall@k on the claude subset is biased upward**: the annotator saw the candidate list
  before deciding.

Every label records its `labeler` and a `confidence` flag (215 high / 85 low).

**This section used to end with "to strengthen this, a human should relabel ~30 strings blind."
That was done.** Sections 3–6 are the result, and it did not go the way the caveat implied.

**The caveat now has a number attached.** Three independent annotators over 120 strings give
**Krippendorff's α = 0.721 [0.651, 0.787]** (§9) — below the ~0.8 publication bar, and the
interval straddles the ~0.67 floor rather than clearing it. Quote the interval alongside any
entity-level figure: it is the measured version of everything above.

## 2. Candidate generation (2.4)

8,187 USDA entities → ~78 candidates per ingredient string, by exact chunked cosine (no ANN)
plus head-noun token blocking. `recall@k` is the **ceiling on everything downstream**: a scorer
cannot pick an entity that was never shown to it.

| k | recall |
|---|---|
| 5 | 68.6% |
| 10 | 75.0% |
| 25 | 84.8% |
| **50 (generation k)** | **90.5%** |
| 100 | 92.0% |

Null class excluded from the denominator: 36 of 300 strings (12.0%) have no reasonable USDA
match and are labeled `no-match`. They are a real answer — `bisquick`, `kitchen bouquet`,
`old bay seasoning`, `orange kool-aid` genuinely are not in Foundation + SR Legacy.

### By stratum

| Stratum | n | recall@50 |
|---|---|---|
| head (≥20 occurrences) | 92 | 93.5% |
| mid (3–19) | 88 | 86.4% |
| tail (1–2) | 84 | 91.7% |

Mid scores *worst*, not the tail — the opposite of the usual expectation. The tail is full of
brand names and parser artifacts that are cleanly `no-match` (excluded from the denominator),
while the mid stratum holds real but awkward foods (`kitchen bouquet`, `rosamarina macaroni`)
that do have candidates but retrieve poorly.

### By generator, each running alone

| Generator | recall@50 |
|---|---|
| `embed_search` (facets reversed: "raw whole egg") | 90.5% |
| `embed_desc` (raw description: "Egg, whole, raw") | 88.6% |
| `token_head` (head-noun block) | 40.2% |

**The A/B is close.** Reversing the facets is worth ~2pp, not the large gain assumed when the
two text forms were introduced. Token blocking alone is weak but cheap, and a genuine
complement: it retrieves by exact head-noun match where embeddings drift semantically
(`soda` → soft drinks rather than baking soda).

> **Measurement bug found and fixed here.** The first version recorded a single `method` per
> candidate — whichever generator found it *first*. Since `embed_search` was consulted first it
> claimed nearly every row, making the other two look useless (`embed_desc` 1.1%, `token_head`
> 0.4%) and the A/B unmeasurable. Membership is now three independent booleans.

### By label provenance

| Labeler | n | @5 | @25 | @50 |
|---|---|---|---|---|
| human | 17 | 88.2% | 94.1% | 94.1% |
| claude | 247 | 67.2% | 84.2% | 90.3% |

The human subset scores *higher*, which is reassuring — rubber-stamped labels would have put
the LLM subset near 100%. n=17; a smoke test, not a validation.

> **Rule-6 duplicate credit: implemented, and it changes recall by nothing.** The guide credits
> any entity sharing the gold `description_raw` (94 descriptions exist twice, Foundation + SR
> Legacy — 188 ids). Applying it leaves every figure above *identical*. 21 of the 264 resolvable
> gold entities have a twin, and in **zero** cases does the twin appear in top-k without the gold:
> identical descriptions produce identical embeddings, so twins land at adjacent ranks and are
> either both retrieved or both missed. An earlier version of this doc predicted these figures
> were understated; they were not. The credit still matters for **2.5**, where the scorer picks a
> single entity and raw id equality would mis-score a twin pick.

## 3. Blind relabel, pass 1 — 36% agreement

30 strings drawn uniformly from the 282 annotator-labeled ones (seed `20260729`, frozen in
`relabel_manifest.json` before judging, population fingerprint `12a538d85d6751a7`). Judged
without the existing label visible; `labels.jsonl` never modified. 25 judged, 5 skipped.

**Agreement: 9/25 = 36.0% (95% Wilson CI 20.2–55.5%).**

| Original confidence | n | agreement |
|---|---|---|
| high | 17 | 47.1% |
| low | 8 | 12.5% |

The annotator's own confidence flag is informative — low-confidence labels agree far less.

Not an anchoring artifact: the human took the top-displayed candidate on only 11 of 25 rows
(44%) and never used search.

## 4. Adjudication of the 16 disagreements

Each disagreement re-presented as **A/B in randomized order, neither attributed**, asking not
"which do you prefer" but "which one does the guide select". Order seeded per string; every
record keeps `shown_first` so the mapping is auditable.

| Verdict | n | Meaning |
|---|---|---|
| guide selects the gold label | 9 | the relabel was wrong |
| guide selects the relabel | 7 | **the gold label is wrong** |
| genuinely ambiguous | 0 | — |

**The blinding worked.** The judge overturned their *own* pass-1 choice on 9 of 16 (56%). Had
this been self-defense in disguise it would have come back near 16/16 for the relabel.

**Gold labels defensible: 18/25 = 72.0% (CI 52.4–85.7%)** → an estimated **28% error rate**
(CI 14.3–47.6%). Even the optimistic end of that interval is far above the 5% error budget
implied by the plan's ≥95% precision target.

## 5. What the disagreements were actually about

They were not random annotator noise. **Rule 1 ("prefer the least-qualified base form") assumes
a least-qualified entry exists**, and across much of USDA it does not: every milk states a fat
level, every pasta enriched or unenriched, every green bean raw/canned/frozen. With no tiebreak
written down, two labelers broke it differently and *consistently* — the relabel systematically
chose the more-qualified entry.

5 of the 7 rejected labels were exactly this. 2 more were the `no-match` boundary
(`cracked peach pit`, `sorrel and chervil`).

So the guide gained **rule 2b (culinary default: full-fat, enriched, raw when the line is
silent)** and **rule 4b (a part or derivative USDA does not carry is `no-match`; a line naming
several foods takes the dominant one)**.

> **Independent corroboration.** The [Epicure paper](https://arxiv.org/pdf/2604.22776) builds a
> USDA FDC matching pipeline and breaks ties with a preparation-state preference order —
> raw > fresh > dried > cooked > canned > frozen. Rule 2b is that rule, arrived at
> independently. The convention is standard practice, not a local invention.

### ⚠️ Why the improved number is not evidence

Rescoring the same 25 strings under rules 2b/4b moves gold-defensible from 72.0% to
**88.0% (CI 70.0–95.8%)**, i.e. a 12% error rate. **Do not quote that as a measurement.** The
convention was chosen *after* seeing the disagreements it resolves, and the options presented to
the decision-maker showed which cases each rule would vindicate. That is a rule scored on its
own training set.

Neither number is trustworthy on its own: 72% was measured with no tiebreak available to either
labeler, and 88% was measured with the answers visible. An out-of-sample pass under the amended
guide is drawn and frozen (`relabel_manifest2.json`, 25 strings, seed `20260730`, zero overlap
with pass 1) but **not run** — the project owner declined further hand-labeling, and the metric
moved to nutrition error instead. The machinery remains if that decision is revisited:
`uv run python -m pantryiq.er.relabel --pass 2`.

### Rule 2b audit over the whole gold set

Automated, no human time: **13 of 264 resolved labels flagged (4.9%)** — 11 form, 2 fat;
11 claude, 2 human. Flagged only where the labeler faced a real fork (another entry for the
same food declaring fewer of the silent attributes), so `Spices, chervil, dried` is not flagged
— USDA carries no raw chervil.

It flags; it does not repair. An earlier version also proposed replacements and picked badly
enough to record: `Beans, liquid from stewed kidney beans` for `kidney bean`, `Bread, wheat` for
`corn bread`. Roughly 3 of the 13 are false positives on inspection (`condensed cream of chicken
soup` — canned is implied by *condensed*; `karo syrup` — "light" is a corn-syrup grade, not a
fat level).

## 6. Nutrition error — the reporting unit

Nothing downstream consumes an `fdc_id`; it consumes kcal per 100g. Relative gap denominated by
the **larger** of the two values (symmetric, bounded at 100%; relative-to-gold reads a 32-vs-254
kcal gap as 694%, which says nothing useful about a food that is mostly water), with an absolute
floor of 5 kcal/100g.

Pass 1's disagreements, priced:

| Relative kcal error | pairs |
|---|---|
| within 10% (nutritionally equivalent) | 13 / 22 |
| 10–25% | 0 |
| 25–50% | 6 |
| over 50% (materially wrong) | 3 |

**Nutritional agreement: 13/22 = 59.1% (CI 38.7–76.7%)**, against 36.0% strict entity
agreement. Plus 3 null-class disagreements, which are a coverage failure rather than a magnitude
error.

This is **not** a rescue — 9 of the 13 comparable disagreements were real, 3 of them >50%. What
it does is stop charging full price for disputes that cost nothing: `spiral pasta`
(enriched vs unenriched) is a **0.0%** kcal difference, `rounded tbsp flour` 0.5%,
`pineapple juice` 5.7%, `whole tomato` 3 kcal.

## The ambiguity ceiling

The kcal spread among entities naming the **same food** as the gold label. Needs no labels, and
it bounds what any precision claim can mean.

| Spread within the correct food | strings |
|---|---|
| within 10% | 50 / 163 |
| 10–25% | 21 / 163 |
| 25–50% | 25 / 163 |
| **over 50%** | **67 / 163** |

**Median 31.6%**, measurable on 163 of the 264 resolved labels.

`pinto bean` spans 82 kcal/100g canned to 333 dry. `beef bouillon cube` spans 3 reconstituted to
170 dry. `clams with liquid` spans 2 to 202.

**A resolver that identifies the food perfectly every time is still this far off on nutrition,
because the ingredient line does not say which facet it means.** That is under-specification in
the recipe text, not a modelling failure — and it is why the plan's ≥95% precision target was
never reachable for reasons unrelated to the scorer.

Read as an **upper** bound: "same food" is the two leading facets, which is right for
`('beans','pinto')` but too coarse for `('beverages','tea')`, where it groups 33 unrelated teas.
10 of the 163 are flagged over-grouped.

> **Defect found by inspection, not by tests.** The first version reported `Beverages, tea` at
> 100% ambiguity: the group spans 0 to 1 kcal/100g, and `|0-1|/1 = 100%`. Any group whose
> minimum is 0 reported 100% regardless of how trivial the gap. Hence the absolute floor.

## Why there is no external gold set to use instead

Checked, because the labeling cost prompted the question. Human-annotated recipe corpora exist —
[FoodBase](https://academic.oup.com/database/article/doi/10.1093/database/baz121/5611291)
(1,000 Allrecipes recipes, 12,844 annotations, 2,105 unique entities) and
[CafeteriaFCD](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9455825/) — but they link entities
to **Hansard, FoodOn, SNOMED-CT and FoodEx2**: food *ontologies*, not a nutrition database.
[FoodSEM](https://arxiv.org/abs/2509.22125), the current SOTA model for food NEL, is evaluated
on the same ontologies. No public dataset links free-text recipe ingredient strings to USDA FDC
ids as human ground truth.

That absence is the reason this problem is worth demonstrating. FoodBase remains usable as an
external check on the *parser/NER* layer, which is a real slice of the pipeline scored against
someone else's annotations.

## 7. Scoring and routing (2.5) — the frozen holdout, read once

Everything fitted on the 201-string tune split and applied unchanged to the 99-string holdout:
eight features → logistic regression → isotonic calibration → thresholds. Regenerate:
`uv run python -m pantryiq.er.report`.

### Headline

| | holdout |
|---|---|
| **nutritionally equivalent (<10% kcal, or <5 kcal/100g)** | **54/79 = 68.4%** (95% CI 58.2–78.5%) |
| median relative kcal error | **0.0%** (CI 0.0–2.7%) |
| mean relative kcal error | 16.7% (CI 10.9–23.4%) |
| entity top-1, of strings whose gold was retrieved | 48.1% |
| entity top-1, of all 99 holdout strings | 38.4% |
| retrieval ceiling on this split | 90.8% |

| error band | n |
|---|---|
| within 10% (nutritionally equivalent) | 54 |
| 10–25% | 6 |
| 25–50% | 7 |
| over 50% (materially wrong) | 12 |

By stratum: head 70.0% (CI 53.3–86.7), **mid 59.3%** (40.7–77.8), tail 77.3% (59.1–95.5). Mid is
worst, the same inversion recall@k shows.

**Read against the ceiling, not against 100%:** the ambiguity ceiling on these same strings is a
median **24.3%** kcal spread *within* the correct food.

### The learned scorer does not beat top-cosine

| split | difference in kcal-equivalence (model − baseline) | McNemar |
|---|---|---|
| tune (out-of-fold) | **+3.8pp** [−0.6, +8.2] | 4 / 10, p=0.18 |
| **holdout** | **+0.0pp** [−7.6, +7.6] | 5 / 5, p=1.000 |
| holdout, entity top-1 | **−2.5pp** [−10.1, +5.1] | 6 / 4, p=0.754 |

The holdout row was **−3.9pp before the tie-break fix** (§13). Candidate `rank` was decided by
Python `set` iteration order on tied cosines, and the baseline arm is defined as `rank = 0`, so
the comparison was partly measuring hash layout. With an explicit tie-break the two methods are
*exactly* level on nutrition — 5 wins each — which states the same conclusion more cleanly.

**The gain does not survive out of sample** — which is what a noise difference does. The paired test
at step 3b said "cannot distinguish"; the adoption decision over-read a consistent point-estimate
direction as weak evidence, and the holdout corrected it. Eight engineered features, a logistic
model and isotonic calibration are worth *nothing measurable* over taking the top embedding
cosine.

Switching to the baseline *because the holdout prefers it* would be selection on the holdout and
would void the 68.4% estimate. So that figure stands as the committed pipeline's number. The
defensible argument for shipping the simpler resolver is that the two are indistinguishable on
**both** splits, so simplicity breaks the tie — and after the §13 tie-break fix they are exactly
level on nutrition, which makes the point without needing the argument.

### Routing: no confident subset, but usable failure detection

On tune, the coverage/accuracy curve is flat-to-inverted at the top — the most confident 10% of
strings scored 50%, *below* the 58.2% base rate, and the largest band meeting the plan's 90%
target held 1.7% of strings. Raw `p(best)` correlates with nutritional equivalence at Spearman
+0.13, non-monotonic by quintile.

The cause is structural: `p(best)` is driven by embedding cosine, and a high cosine says the
**food** is right. Nutritional equivalence turns on the **facet** — canned or raw, whole or
nonfat — whose descriptions are nearly identical, so cosine barely separates them. Same finding
as the ambiguity ceiling, reached from the model side.

Out of sample it is better than that suggests, in one direction only:

| calibrated band | n | kcal-equivalent |
|---|---|---|
| high (≥0.69) | 44 | 68.2% |
| mid (0.51–0.69) | 30 | 60.0% |
| **low (<0.51)** | **11** | **9.1%** |

It cannot certify correctness; it can flag likely-wrong picks. **Consequence for the plan: 2.6's
Claude adjudicator is required, not an optimization** — there is no confident subset to skip it
on, and the brief's "% resolved without an LLM call" has the answer *essentially none, at any
defensible accuracy target*.

### Anchoring — RETRACTED, the treatment was never administered

This section used to report a "clean null" from a control group shown scrambled, unranked
candidates:

| presentation | n | kcal-equivalent |
|---|---|---|
| ~~control (unranked, no suggestion)~~ | ~~7~~ | ~~57.1%~~ |
| ~~ranked~~ | ~~78~~ | ~~57.7%~~ |

**That comparison was vacuous.** The scrambled-candidate control exists only in the interactive
labeling CLI, and 282 of 300 labels came from an LLM annotator that never touched it — **40 of
the 42 control-designated rows were annotator-produced**, so exactly one string in the entire gold
set (`buttermilk`) was ever actually presented unranked. The two arms were random subsets of
*identically produced* labels, and the null was guaranteed by construction. The problem is not
low power; it is that the intervention never happened.

`docs/labeling_guide.md` claimed "~15% of rows (42 of 300) are shown scrambled and unranked" —
false for 40 of the 42, now corrected there.

**The project's real anchoring control is §9's ensemble**, which shuffles candidate order per
annotator and records `shown_order` on every judgment. Read that instead.

## 8. The shipped resolver, and why "mid is weak" was the wrong question

### What ships

`uv run python -m pantryiq.er.resolve` — **the top embedding cosine, and nothing else**, plus a
confidence flag from an isotonic curve fitted on tune (`data/confidence_curve.json`, so resolving
is a lookup rather than a fit). `features.py` / `scoring.py` / `routing.py` remain as the
reproducible record of the experiment in §7; **nothing in the shipped path imports them.**

Holdout: **64.4% nutritionally equivalent** over all 87 resolvable strings, median error 0.0%.
Note the denominator differs from §7's 63.6% (n=77), which excluded strings whose gold was never
retrieved; 87 is the harsher and more honest count. Both come from the same single holdout read.

The flag separates usefully out of sample: 18 strings flagged at 44.4% equivalent against 69.6%
for the rest. The cut (0.40) was chosen on tune — the isotonic curve is a coarse step function, so
anything in 0.40–0.50 flags the same 23% of tune strings.

> **Defect caught by a test:** the resolver returned identical picks in a *different row order* on
> successive calls — no outer `ORDER BY`. Harmless for correctness, but it would have made every
> run-to-run diff of the resolved output pure noise.

### Frequency stratum is not the real axis — label confidence is

The holdout showed mid at 50% against head 67% and tail 77%, which looked like the strongest lead
available. Over all 300 labels the gap is smaller (head 60 / mid 54 / tail 64), and conditioning
on label confidence dissolves most of it:

| stratum | high-confidence labels | low-confidence labels |
|---|---|---|
| head | 65% (n=77) | 31% (n=13) |
| mid | **62%** (n=55) | 41% (n=32) |
| tail | 81% (n=54) | 31% (n=29) |

**Among high-confidence labels, mid (62%) matches head (65%).** What mid actually has is more
uncertain labels — 37% flagged low-confidence against head's 13%. And low-confidence labels score
31–41% in *every* stratum, a ~30pp effect that dwarfs any difference between strata.

Since pass 1 measured low-confidence labels agreeing with a blind human relabel only **12.5%** of
the time (against 47.1% for high-confidence), a substantial part of what these numbers call
resolver error is **label error**. The 85 low-confidence labels are the highest-value target in
the project, and the agreed multi-annotator ensemble is exactly the instrument for them.

Mid also has the *lowest* irreducible ambiguity (median 22% kcal spread within the correct food,
against head's 49%), which rules out facet ambiguity as the explanation.

### Two real error patterns in the failures

- **Dry mix vs ready-to-eat**, and it is expensive: `chocolate pudding` gold *dry mix* 378 kcal vs
  picked *ready-to-eat* 142; `black cherry jello` gold *dry mix* 381 vs picked *cherry juice* 59;
  `coffee creamer` gold *powdered* 529 vs picked *fluid* 195. Recipes naming a pudding or jello
  almost always mean the box. This is the same form-ambiguity family as rule 2b and is not yet
  covered by it.
- **Lexical traps on short strings**: `fettucine` → `Cheese, feta`, `kraut` → `Kohlrabi, raw`,
  `cider` → `Vinegar, cider`, `bacon slice` → `Bacon, meatless`.

### A fix that was measured and rejected

The resolver ignores `is_deprioritized`, so babyfood and brand entries can win — `fettuccine
noodle` → `Babyfood, dinner, beef noodle, junior` — which guide rule 3 forbids. Skipping
deprioritized candidates outright: **56.6% → 57.0% on tune, McNemar 1 vs 1, p = 1.0.** Only 4% of
picks are affected, and a blanket skip *breaks* the strings that name the brand: `jimmy dean
sausage` went from the correct JIMMY DEAN entry to `Sausage, meatless`, `peanut m m s` from M&M's
to `Peanuts, raw`. Rule 3 is conditional on the line naming the brand, and 7 affected strings
cannot validate brand-matching logic. Not adopted.

## 9. Multi-annotator ensemble — α = 0.709, and it does not beat the gold labels

The instrument aimed at §8's finding that low-confidence labels are the project's weakest link.
Three **independent** annotators (`claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`) judge
each string; the majority vote is the candidate label, and the spread between annotators is the
quality measure. 120 strings × 3 = **360 judgments, 0 failures**. Regenerate:
`uv run python -m pantryiq.er.ensemble --report`.

Four design choices make the number mean something:

- **Three different models, not one model three times.** At these settings the same model returns
  near-identical answers, so an "ensemble" built that way measures nothing.
- **Candidates shuffled per string** (deterministic by seed), *not* in `display_score` order — the
  original 282 labels were produced while seeing that ranking, and reusing it would re-import the
  anchoring §7 *tried* to measure and could not (that diagnostic is retracted; this shuffle is the
  project's only working anchoring control). `shown_order` is recorded on every judgment.
- **The rules are sliced out of `docs/labeling_guide.md` at runtime**, not restated in the prompt.
  Restating them is precisely the drift that produced §3's 36% disagreement.
- **Recipe lines are treated as data**, wrapped in a tag with an explicit instruction to ignore
  directives inside them (the brief's injection-hardening requirement).

### The agreement number

**Krippendorff's α (nominal) = 0.721, 95% CI [0.651, 0.787]** (5,000-item bootstrap) over all 120
strings; **0.780** over the 25 mixed-confidence validation strings. Unanimous on **76/120
(63.3%)**. **9 strings** were three-way
splits — three capable models, three different answers, no majority at all.

0.721 sits below the ~0.8 publication bar, and its interval [0.651, 0.787] **straddles** the ~0.67
floor rather than clearing it — on 120 items there is roughly a 1-in-8 chance the labels fall in
the band conventionally called too low to draw conclusions from. **It confirms the §1 caveat
rather than dissolving it:** these labels support directional conclusions, not precise ones. That
was already the doc's prose position; it is now a statistic a reader can interpret — provided the
interval is quoted with it.

| Annotator | agreement with the majority vote |
|---|---|
| `claude-sonnet-5` | 98.2% (the median annotator) |
| `claude-opus-5` | 87.4% |
| `claude-haiku-4-5` | 82.9% |

### It is not an improvement, and aggregation actively hurt

Scored against the **25 independent human judgments** — the pass-1 blind relabels, where the human
verdict was formed without seeing the gold label:

| | n | entity | kcal-equivalent |
|---|---|---|---|
| `claude-opus-5` alone | 25 | **48.0%** | 65.2% [45–81] |
| `claude-haiku-4-5` alone | 25 | 44.0% | **68.2%** [47–84] |
| `claude-sonnet-5` alone | 25 | 36.0% | 59.1% [39–77] |
| **majority vote** | 23 | 39.1% | 61.9% [41–79] |
| **current gold label** | 23 | 34.8% | 60.0% [39–78] |

The vote beats the existing labels by +4.3pp entity and +1.9pp kcal — entirely inside the noise at
n=23. Worse for the premise: **the majority vote scores below the best single annotator** (39.1%
vs 48.0% entity). The two weaker models outvoted the strongest, which is the opposite of what
ensembling is supposed to do. Selecting `claude-opus-5` on the strength of that table would be
selection on a 25-row validation set — the same error §7 records — so it is noted, not acted on.

> ⚠️ **Measurement bug caught before this was reported.** The first run showed the gold labels at
> 61.5% entity / 77.1% kcal, apparently beating the ensemble by 15pp. That was an artifact: 18 of
> the 43 human-judged strings are ones a human labeled *directly*, so for those the gold label
> **is** the human judgment and scored 100% for free. Only the 25 blind relabels are independent.
> `report()` now excludes the self-scoring rows and prints how many it dropped.

**Decision: the votes were not promoted.** `labels.jsonl` is unchanged. The ensemble did the job
it was built for — measuring label quality — and the measurement says overwriting 85 labels with a
statistically indistinguishable aggregation would buy churn, not accuracy.

### What this implies for 2.6

2.6's premise is that Claude adjudication *fixes* the resolver's mistakes. Put the two side by
side:

| | entity | kcal-equivalent |
|---|---|---|
| shipped resolver — top cosine (§7 holdout) | 48.1% | 68.4% |
| `claude-opus-5` with the full guide + 40 candidates | 48.0% | 65.2% |

**An LLM given the complete labeling guide and 40 candidates performs about the same as taking the
top embedding cosine.** Different string sets and small n, so this is suggestive rather than
settled — but it is the first direct evidence on the question, and combined with aggregation
making things worse it moves the prior on 2.6 substantially. Worth measuring properly (one
annotator over the 201-string tune split, paired against the resolver on identical strings using
the `mcnemar` / `paired_bootstrap` machinery) **before** building 2.6 rather than after.

Two incidental findings worth keeping: `claude-haiku-4-5` cannot use the prompt cache here — its
minimum cacheable prefix is 4,096 tokens and the rules prefix is ~1,800, so all of its input bills
at full rate while the other two cache theirs. And every judgment records which guide rule the
annotator applied, giving a per-rule audit trail of what actually decided each call.

## 10. Is 2.6 worth building? — the question is structurally unanswerable here

Built to decide whether to write 2.6's Claude adjudicator, using `claude-opus-5` (the strongest
annotator in §9) paired against the shipped resolver on the 201-string tune split. Regenerate:
`uv run python -m pantryiq.er.adjudicator_value --report`.

**It returns a non-answer, and that is the finding.**

### The test's own premise was wrong

It rested on a paired-design argument: both arms are scored against the same contested gold, so
shared label error cancels in the difference. That holds only when the error is *shared*. Here it
is **correlated with one arm**. 166 of the 177 comparable tune labels were produced by a Claude
annotator following this same guide, so a Claude adjudicator re-running that process agrees with
them because it *is* the process that made them — while the resolver uses embeddings, an
independent method, and necessarily looks worse.

| vs annotator-produced gold (n=177) | resolver | adjudicator | difference |
|---|---|---|---|
| entity top-1 | 36.2% | 78.5% | **+42.4pp** [+35.0, +49.7], p < 0.0001 |
| kcal-equivalent | 59.1% | 92.1% | **+32.9pp** [+26.2, +40.2], p < 0.0001 |

**Those numbers measure self-consistency, not correctness.** §1 stated the constraint before any
of this was built: *"A Claude adjudicator cannot be honestly scored against these labels."* The
trap was documented and walked into anyway, which is why the invalid figures are printed here
rather than deleted — a reader who tries the same test should recognise the shape of the result.

### The valid comparison has no power

Against the blind human relabels — the only independent reference — the tune split overlaps just
**11** strings (9 with kcal on both sides):

| vs independent human judgment (n=11) | resolver | adjudicator | difference | p |
|---|---|---|---|---|
| entity top-1 | 27.3% | 45.5% | +18.2pp [−18.2, +54.5] | 0.63 |
| kcal-equivalent | 33.3% | 44.4% | +11.1pp [−22.2, +44.4] | 1.00 |

**Underpowered, not negative.** At n=11 no effect of any plausible size would reach significance;
this is not evidence that an adjudicator fails to help.

> **A second bug, caught after the first.** `compare()` scored every section against the gold
> label, so filtering rows to those *with* a human judgment left the reference standard unchanged
> — the section headed "the only valid comparison" was printing self-consistency on a subset
> (+54.5pp instead of +18.2pp). The reference is now an explicit argument, and a regression test
> flips it on identical rows to prove opposite verdicts come out.

### What is measurable without ground truth

The resolver and the adjudicator are **genuinely independent methods** — embeddings versus an LLM
reading the guide. They agree on only **69/177 (39.0%)** of tune strings, and **11/41 (26.8%)**
inside the resolver's own low-confidence band.

That disagreement flags contested strings *without needing to know which arm is right*. It is the
one use of an adjudicator this label set can support, and it suits the product better than "the
LLM fixes the answer" did: §7's ambiguity ceiling already showed that a large share of these
strings are under-determined by the recipe text, and a flag is the honest response to that.

### Conclusion

**2.6's accuracy is structurally unmeasurable against the current gold set** — not merely
unmeasured. Resolving it needs more independent human judgments, which the project has ruled out
(see the working agreement). Build 2.6 if desired, but it **cannot be claimed to improve
accuracy** on this evidence. Building it as a *disagreement flag* rather than a corrector is
defensible today and needs no reference labels.

## 11. The entity map and the corpus headline (2.7)

`silver.ingredient_entity_map` — every distinct ingredient string resolved to a USDA entity with
its nutrition, a confidence score and flag, and the occurrence count that says how much it
matters. Regenerate: `uv run python -m pantryiq.er.entity_map`.

| | |
|---|---|
| strings with an entity-map row | **9,163** |
| — the resolver commits to an entity | 6,279 (68.5%) |
| — the resolver declines (`abstained`) | **2,884 (31.5%)** · 14,063 occurrences (**12.5%**) |
| ingredient-line occurrences covered | **112,452 / 112,463** |
| with an energy value | 8,949 strings (97.7%) · 109,235 occurrences (97.1%) |
| flagged low-confidence | 2,727 strings (29.8%) · 13,672 occurrences (**12.2%**) |

The 11 uncovered lines are ones whose parser output was empty — there is no string to resolve.

### The headline is a stratified estimate, not a sample average

The gold set was drawn 100/100/100 from strata now holding **543 / 1,558 / 7,062** distinct
strings, so the tail is over-sampled by roughly 70× relative to the head. Averaging the 300 labels
directly answers "how does the pipeline do on the gold set", which is not a question anyone has.
Two reweightings answer the two real ones:

| | per unique string | per occurrence |
|---|---|---|
| **nutritionally equivalent (<10% kcal, or <5 kcal/100g)** | 69.2% [60.1, 78.0] | **77.4%** [62.7, 86.4] |
| entity top-1 | 47.5% [37.8, 57.5] | 58.1% [36.7, 73.2] |

By stratum (unweighted, for reference): equivalence head 69.7%, mid 52.3%, tail 72.9%.

**These are conditional on the resolver not declining** (§12) — it abstains on 31.5% of strings
and 12.5% of occurrences. Accuracy and coverage move in opposite directions by construction, so
neither figure is quotable alone.

**The equivalence band has an absolute floor as well as a relative one**: a pick counts if it is
within 10% *or* within 5 kcal/100g. The floor stops near-zero foods (`Beverages, tea`, 0 vs
1 kcal/100g) from registering as 100% errors, but it is not free — it is worth **+2.7pp per
unique string and +0.5pp per occurrence**, and it credits a handful of picks at 18–28% relative
error. Reported rather than folded into the headline:

| floor | per unique string | per occurrence |
|---|---|---|
| **5 kcal (shipped)** | **69.2%** | **77.4%** |
| 2 kcal — enough for the tea case alone | 67.8% | 77.0% |
| none (pure relative error) | 66.5% | 76.9% |

**The per-occurrence figure is the one that describes the product** — it weights each sampled
string by how often it actually appears, and head strings carry 83.8% of all occurrences. It is
~8pp better than the per-string number for exactly that reason. Its interval is wide because the
weighting concentrates on a handful of very frequent strings — the head stratum's Kish effective
sample size is only ~11, with one string carrying a quarter of the stratum's weight. That is a
real limit of a 300-label sample, reported rather than smoothed; if anything the true interval is
wider than the bootstrap's.

**Both figures inherit the gold labels' quality.** α = 0.721 [0.651, 0.787] (§9). They estimate
agreement with those labels, not with ground truth, and the tooling prints that caveat next to
the numbers.

### Two findings from building it

- **Flagged strings are 29.8% of the vocabulary but only 12.2% of occurrences.** The hard strings
  are overwhelmingly rare ones, so the experience of using the map is better than the
  vocabulary-level numbers suggest. This is the same head/tail asymmetry that makes the two
  denominators diverge, showing up in coverage instead of accuracy.
- **The confidence flag detects the null class**, which §7 never tested: **63.9%** of gold
  `no-match` strings are flagged against **22.3%** of resolvable ones — despite the threshold
  having been fitted for nutrition error, not null detection.

> **This section originally ended with a limitation: the resolver could not emit `no-match`, so
> the ~12% of strings with no real USDA entity were silently assigned one.** That is fixed — see
> §12.

## 12. A fitted `no-match` threshold — letting the resolver decline

§11's limitation was that the resolver always returned its best candidate, so a string with no
real USDA entity got one anyway, producing confident wrong nutrition. Regenerate:
`uv run python -m pantryiq.er.abstain`.

### The signal is raw cosine, not the confidence curve

Measured on tune, by how well each separates the null class (AUC — the probability a resolvable
string outscores a `no-match` one):

| signal | AUC | |
|---|---|---|
| **raw top-1 cosine** | **0.771** | |
| calibrated confidence (fitted for *nutrition error*) | 0.764 | *not a real comparison — see below* |
| top-1 / top-2 margin | 0.598 | the measured one |

> **Correction (2026-07-31).** This section used to present all three rows as measurements, with
> cosine "beating" the confidence curve. **Cosine vs the curve is an identity, not a result.** The
> curve is an *isotonic* fit of cosine, therefore a monotone transform of it, and a monotone
> transform cannot rank the null class better — only coarser. The stored curve has 8 distinct
> output values, and the whole 0.007 AUC gap is ties scored at 0.5. Cosine is used because it is
> continuous, not because it carries more signal.

**The margin comparison is the real finding.** Margin is the classic abstention signal, and it is
nearly useless here for a reason specific to this problem: a margin is small when two candidates
*compete*, but a `no-match` string has no good candidate at all — its whole candidate set scores
low together. Absolute similarity carries the signal; relative similarity does not.

§11 reported that the old flag caught 63.9% of the null class. That was the nutrition-fitted
curve doing null detection by accident; this threshold is fitted for the job.

### The objective is deliberately not F1

Guide rule 4 states the asymmetry: *"a wrong match is worse than an honest gap, because a wrong
match silently produces wrong nutrition downstream."* So the threshold maximizes **F2** on the
null class — catching a null counts twice as much as avoiding a false abstention. β was fixed
from that rule **before** the downstream effects below were measured.

| rule | threshold | abstains (strings / occurrences) | null recall (CV) | resolved purity | nutrition among resolved |
|---|---|---|---|---|---|
| none | — | — | — | 88.1% | 56.6% |
| F1 | 0.542 | 15.4% CV | 41.7% | 92.2% | 56.5% |
| **F2 (shipped)** | **0.644** | 31.5% / **12.5%** corpus | **62.5%** | **95.1%** | **62.3%** |

**Three different denominators sit in that table, and only one column is cross-validated.** Null
recall is out-of-fold (the threshold is genuinely refitted inside each fold, so it is what a newly
fitted cut achieves on unseen strings). `resolved purity` and `nutrition among resolved` are
**in-sample** on the split the threshold was fitted on. The F2 abstain rate is corpus-wide across
all 9,163 strings; the F1 row's is the CV rate on tune. They are not on one scale.

Applying the shipped threshold to the untouched holdout gives the honest out-of-sample read:
null recall **58.3%** against tune's in-sample 70.8%, with CV at 62.5% in between — which is what
an honest cross-validation should look like.

Two things decide it. **The occurrence view**: declining on 31.5% of the vocabulary costs only
12.5% of what a user actually hits, the same head/tail asymmetry that runs through §11. And **F1
buys nothing** — it improves purity but leaves nutrition among the resolved unchanged (56.6% →
56.5%), while F2 gains +5.7pp.

Null-class precision at F2 is **21.1% cross-validated** (26.9% applying the shipped cut to the
holdout): roughly four of every five abstentions are on strings that *do* have a valid entity.
That is the price rule 4 asks you to pay, stated plainly rather than buried in an F-score. The
figure also moves with the CV seed — across seeds it spans roughly 0.19–0.24, so treat it as a
range, not a point.

### It is a second, independent signal — not a replacement

`flagged` (isotonic confidence, fitted for nutrition error) means *"this pick may have the wrong
facet"*. `abstained` (raw cosine, fitted for the null class) means *"this string probably has no
USDA entity at all"*. They answer different questions and are not interchangeable. An unfitted
threshold loads as 0.0, so the resolver behaves exactly as before until `abstain.py` has run.

## 13. The review pass (2026-07-31)

Four independent reviews — three local agents (statistical methodology, code correctness, test
quality) and one cloud review — went over the finished Phase 2. Everything below was fixed and
the pipeline re-run end to end. **No gold label was re-judged**; 11 strings were renamed
mechanically (see below).

### What was wrong

| # | Finding | Effect |
|---|---|---|
| 1 | **`adjudicator_value.py` crashed** — unpacked 5 values from a `resolve()` that returns 6 since §12. Found by 3 of the 4 reviewers. | §10 was unregenerable. Its test stubbed `resolve` with a 5-tuple, so the suite stayed green while the module could not run at all. |
| 2 | **Candidate `rank` broke ties on `set` iteration order** — a hash-layout detail. 5.2% of strings tie at top cosine. | `rank = 0` *is* the §7 baseline arm, so the headline negative result partly measured hash order. |
| 3 | **Five parser artifacts.** `to` missing from `FILLER` left `salt and pepper to` (479 occurrences, head stratum) resolving confidently to tomato chili sauce at 92 kcal/100g. Plus non-idempotent `normalize`, orphaned hyphens (`-inch`), `cookies → cooky`, and `of` surviving the unit strip. | 788 occurrences confidently mis-resolved; 23 strings where `normalize(normalize(x)) ≠ normalize(x)`. |
| 4 | **The §7 anchoring diagnostic measured nothing** — the control was never administered to 40 of its 42 rows. | Retracted; see §7. |
| 5 | **§12's cosine-vs-curve AUC comparison is an identity**, not a measurement — isotonic is monotone. | Reframed; the margin comparison is the real result. |
| 6 | **α ignored rule-6 twins** in `majority`/`alpha`/unanimity while applying it in the same function's agreement table, and was reported with no interval. | α 0.709 → **0.721 [0.651, 0.787]**. |
| 7 | **The 5 kcal floor was undisclosed** and the metric was labelled "within 10%". | Now labelled and its sensitivity published (§11). |

### What changed in the numbers

| | before | after |
|---|---|---|
| distinct strings | 9,324 | **9,163** |
| equivalence, per unique string | 69.0% [59.9, 77.6] | **69.2%** [60.1, 78.0] |
| equivalence, **per occurrence** | 73.4% [53.8, 84.8] | **77.4%** [62.7, 86.4] |
| entity top-1, per occurrence | 54.4% [29.0, 72.1] | **58.1%** [36.7, 73.2] |
| §7 holdout equivalence | 63.6% [53.2, 74.0] | **68.4%** [58.2, 78.5] |
| §7 holdout, model − baseline (kcal) | −3.9pp [−13.0, +3.9] | **+0.0pp** [−7.6, +7.6] |
| Krippendorff's α | 0.709 | **0.721** [0.651, 0.787] |
| recall@50 | 90.5% | 90.5% (unchanged) |

The per-occurrence figure moved most because the parser fixes merged junk *head* strings into
real ones, and the head stratum carries 83.8% of occurrences. **No conclusion in this document
reversed.** The ML still earns nothing — and now reads more cleanly, at exactly level on
nutrition (5 wins each, p=1.000) instead of a noisy −3.9pp.

### The honest caveat on this re-run

The gold sample was drawn from the *old* vocabulary. 289 of 300 strings were unaffected; 11 were
renamed 1:1 into the same food better spelled (`oreo cooky` → `oreo cookie`, `garlic to` →
`garlic`), with no collisions, so the `fdc_id` judgments transferred unchanged. But **5 of those
merged into busier strings and changed frequency stratum**, so the sample is no longer an exact
probability sample of the new vocabulary — the estimator post-stratifies from the live table.
The perturbation is small (5/300) and stated here rather than smoothed over.

The holdout was read again to re-derive §7. That read was for a **correctness fix, not model
selection** — no parameter was chosen on it, which is the property §7 protects.

### Test coverage, which was the weakest part

50 mutations were run against the suite: **14 survived**. The measurement layer — the code that
produces every published number — was the least defended part of the codebase, and two tests
named the property they were meant to check and then failed to check it:

- `test_abstention_is_a_separate_signal_from_the_confidence_flag` could not detect abstention
  reading the calibrated curve instead of raw cosine (§12's whole finding), because both signals
  fell the same side of the cut on both fixture rows.
- `test_cross_validation_refits_inside_each_fold` could not detect fitting on all the data —
  in-sample scores *better*, so it passed the `> 0.8` assertion more easily.

Both are rewritten and verified against the mutations they missed. Added:
`test_the_committed_gold_set_still_produces_the_published_headline` (nothing pinned the headline —
a refactor could have moved it 50pp in silence), a `normalize` idempotence property test, and a
regression pinning `resolve()`'s tuple width at the producer.

**All 14 survivors are now closed. 231 → 284 tests, and the sweep re-run finds zero survivors.**
The remaining twelve were closed by:

- `tests/test_published_constants.py` — pins every constant that defines a published number:
  `EQUIVALENT` (and that `BANDS[0]` agrees with it — they can drift apart independently),
  `MATERIAL_KCAL`, `BETA`, `TOP_K`, `ITERATIONS`, `gold.SEED`, `LOW_CONFIDENCE`, and Wilson's
  `z`. Changing any of them now *requires* editing this file, which is the point: it makes a
  headline change deliberate rather than accidental.
- `tests/test_report.py` — `picks` had no test at all and produces the entire §7 headline. Its
  hand-rolled remap from table-global index to split-local score index is the dangerous part:
  get it wrong and every string is scored against a *different* string's model output, the
  report still prints, and every §7 number is quietly meaningless. The fixture interleaves the
  splits on purpose, because any fixture where global and split-local indices coincide cannot
  test the remap.
- `wilson` is now pinned against a **hand-computed** interval (`wilson(27, 30) = [0.7438,
  0.9654]`, derived from the formula rather than from the code's own output), so the test
  catches formula errors as well as a changed `z`.
- Plus direct tests for `bootstrap_ci`, `recall_at_k`'s rank ordering, `save_curve`'s round-trip
  *at its knots* (the old test passed with `x` and `y` swapped), and `same_entity`'s `None`
  guard.

> **A methodology note that cost a false result.** The first sweep reported all 14 as caught. It
> was wrong: the sandbox omitted `scripts/` and `docs/`, so collection errored on every run and
> every mutation "failed" identically. **A mutation sweep is only meaningful against a control
> run that passes** — without one it measures whether the harness is broken, not whether the
> tests discriminate. The numbers above are from a sweep whose control is green at 284/284.

## 14. Phase 3 — grams, and the accuracy nobody had measured

Phase 3 needed a mass per ingredient line, since every nutrient in `silver.usda_foods` is per
100 g while recipes say "2 cups". Building it surfaced a gap in the project's own standards.

### The asymmetry that started it

| | coverage | accuracy |
|---|---|---|
| entity resolution | measured | **measured exhaustively** — 300 labels, §1-§13 |
| gram conversion | measured (58.9%) | **never measured — not one labeled example** |

Every Gold number is `grams x kcal_per_100g / 100`. Phase 2 proved the second factor and said
nothing about the first, yet recipe nutrition is roughly their product. `gram_accuracy.py` (3.3b)
closes that: 34 (ingredient, unit) pairs covering ~46% of converted lines, checked against
**published reference values with their source recorded** in `data/gram_reference.csv`. Unlike
the ER labels — LLM-produced, α = 0.721 — a reader can verify every row of this ground truth.

**First measurement: 64.1% of converted lines were within 10% of the reference.**

### What it found: the kcal metric is blind to mass errors

The corpus's single most frequent conversion, `sugar` + `cup` on 4,170 lines, read **110 g
against a true 200 g**. Bare "sugar" resolved to *powdered* sugar. §11's kcal headline could not
see this — powdered and granulated sugar carry near-identical energy per 100 g — so a 45% mass
error sat in the head of the corpus, unflagged and unabstained, through all of Phase 2.

It was not alone. Auditing the top 40 head strings by occurrence:

| string | occurrences | resolved to | should be |
|---|---|---|---|
| `egg` | 4,399 | **Eggnog** | Egg, whole, raw |
| `flour` | 2,870 | **Millet flour** | Wheat flour, white, all-purpose |
| `milk` | 2,336 | **Crackers, milk** | Milk, whole |
| `pepper` | 1,329 | **Peppermint, fresh** | Spices, pepper, black |
| `cinnamon` | 1,066 | **Bread, cinnamon** | Spices, cinnamon, ground |
| `sugar` | 5,095 | **Sugars, powdered** | Sugars, granulated |

~19,000 occurrences of confidently-wrong head resolutions, all at high cosine, none flagged.

### A principled rule was tried first, and did not work

Before hand-correcting anything, two penalties were measured on all 9,163 strings: a **whole-word**
requirement (the query's head noun must appear as a whole word — "pepper" is not one in
"Peppermint") and a **derived-form** penalty (a leading facet of Oil/Flour/Syrup the query never
asked for).

| | before | after |
|---|---|---|
| gram accuracy, fixed denominator | 64.0% | **64.0%** |
| occurrence-weighted | 61.4% | **61.4%** |
| kcal-equivalent vs gold labels | 58.9% | 60.9% |

It fixed `pepper` and nothing else — 0.05 penalties cannot close the 0.06-0.09 cosine gaps on
`sugar`, `nutmeg`, `coconut` — while moving 963 strings (10.5%) for no measurable gram gain.
**Not adopted**, on the same reasoning as §8's deprioritization fix.

> Two measurement traps caught while running it, both worth recording. A **free denominator**
> showed 83.1% "after" — pure survivorship, because changed picks lacked portion data and simply
> stopped being scored. A **fixed denominator** excludes exactly the pairs the fix helps most.
> Neither is honest alone. And a **plural bug** (stripping the "s" from the query but not the
> description) sent `onion` to `DENNY'S, onion rings` — a blunt penalty across 9,163 strings does
> collateral damage that a single headline number hides.

### What shipped: bounded, auditable overrides

`data/entity_overrides.csv` — 13 head strings, each recording the entity, what it replaces, and
why, applied in `resolve.load_overrides`. **These are assertions, not learned**, adopted only
after the principled alternative was measured and failed. They are deliberately bounded (nothing
unlisted changes), auditable (every row is checkable against USDA), and the file is optional —
absent, the resolver behaves exactly as before.

An overridden string **keeps its measured cosine**. The override corrects *which* entity is
right and says nothing about how confidently the embedding found it; overwriting the cosine would
launder a hand-assertion into a similarity score and silently move both the flag and the
abstention.

| metric | before | after |
|---|---|---|
| gram accuracy (per occurrence) | 64.1% | **83.2%** |
| gram coverage | 58.9% | **63.9%** |
| kcal equivalence (per occurrence) | 77.4% | **81.0%** |
| entity top-1 (per occurrence) | 58.1% | **67.3%** |

Coverage rose because correct entities carry better portion data: `Egg, whole, raw` has a
per-egg weight, `Eggnog` has a cup weight that never applies to "2 eggs".

### How much of that gain to believe — CORRECTED 2026-07-31

**An earlier revision of this section claimed the kcal and entity gains were "not circular" and
called them "the defensible numbers". That was wrong, and it was the most consequential error in
this document.** A review reconstructed the pre-override resolver and recomputed:

| gold set used | kcal per-occ | entity per-occ |
|---|---|---|
| all 203 scorable strings (as published) | 77.4% → 81.0% (+3.6pp) | 58.1% → 67.3% (+9.2pp) |
| **excluding the 13 overridden strings** | 79.5% → 79.5% (**+0.0pp**) | 64.5% → 64.5% (**+0.0pp**) |

**100% of both gains comes from `cinnamon` and `soda`** — the only two overridden strings in the
gold set — and `entity_overrides.csv` sets each to *exactly* its gold `fdc_id`. Measuring
agreement with a label after setting the prediction equal to that label is circular by
construction. The old defence was about label *provenance* ("a separate annotator, before the
overrides existed"); what makes it circular is that the intervention targeted the labelled item.
The bullet that followed it — "both **agree** with their independently-produced labels" — states
the mechanism and presented it as evidence against itself.

The head weighting makes it sharper: `cinnamon` and `soda` carry 1,748 of the head stratum's
15,983 sample weight, so two strings move a corpus-level metric by 9.2pp. The other 11 overrides
are **unmeasurable** against the gold set, so the measured effect is drawn entirely from the
subset guaranteed to be non-negative. There is no downside risk in that estimator.

Grams are the same shape. 11 of the 34 reference pairs are on overridden strings:

| subset | per pair | per occurrence |
|---|---|---|
| all 34 pairs (as published) | 76.5% | **83.2%** |
| 11 overridden pairs | 90.9% | 92.0% |
| **23 non-overridden pairs** | **69.6%** | **77.6%** |

The non-overridden subset is byte-identical before and after — overrides cannot touch it. So the
independent gram-accuracy gain is **0.0pp**, not "+19.1pp overstates it", and `gram_reference.csv`
and `entity_overrides.csv` encode the same belief for those 11 pairs.

**The overrides were kept anyway, and that is a judgement, not a measurement.** `egg` resolving
to *Eggnog* on 4,399 occurrences and `milk` to *Crackers, milk* on 2,336 are wrong on inspection,
by anyone who reads them, whether or not a metric can see it. What cannot be claimed is that a
measurement demonstrated the improvement. **Quote 77.6% / 69.6% as the independent gram accuracy**;
the 83.2% headline includes hand-corrected pairs and the published 81.0% / 67.3% includes two
contaminated tune strings.

### Still wrong after the overrides

5,076 lines among those checked (16.8%), led by `coconut` + `cup` (218 g against 93 g — the
resolver still picks coconut oil) and a cluster of chopped-vegetable cup weights 14-19% light
(`onion`, `green pepper`, `nut`, `walnut`), where USDA's unqualified cup weight differs from its
"cup, chopped" weight and the recipe line does not say which.

### Coverage is the real Phase-3 limitation

63.9% of lines convert, but a recipe needs **every** line to be complete — so completeness
compounds: **only 4.74% of recipes (711) have full nutrition**, 0.74% (111) have full cost, and
after gating per-serving figures on complete coverage only **118 publish a `kcal_per_serving`
and 14 a `cost_per_serving`**. (An earlier revision quoted 2.6% and "73 of 15,000" — those were
pre-override figures printed next to post-override coverage.) Per-line coverage looks healthy; per-recipe coverage is
brutal. `nutrition_coverage` and `data_trust_score` exist so no total is ever quoted without it.

### The quality gate (3.6), and what "blocks" means

`make gate` runs **`dbt build`**, not `dbt run` followed by `dbt test`. The distinction is the
whole feature: `run` + `test` rebuilds every Gold table from bad data and reports the failure
*afterwards*, while `build` interleaves tests with models so a failure **skips everything
downstream**. That is a gate; the other is an alarm.

20 tests over 5 models: referential integrity (every resolved ingredient must name a real
canonical entity — the brief names this one explicitly), uniqueness of the (recipe, line) grain,
`data_trust_score`/coverages bounded to [0,1], strictly-positive grams and prices, and energy
density bounded at 910 kcal/100 g.

**Thresholds are set to the impossible, not the unusual.** Pure fat is ~902 kcal/100 g, so 910
cannot be exceeded by any mixture. A gate tuned to what merely *looks* surprising would fire on
this corpus's genuine 40 lb of pork fat and 13 gallons of ice cream, and a gate that fires on
real data gets muted — which protects nothing.

**Proven, not asserted.** `make prove-gate` injects one impossible row (50,000 kcal/100 g) into
Silver, rebuilds, and checks two things: the test fails, *and* downstream models are skipped.

| | result |
|---|---|
| clean build first | PASS=25 — so a later failure is the probe, not pre-existing rot |
| test detects the bad row | **FAIL** on `accepted_range_canonical_ingredients_kcal_per_100g` |
| Gold models rebuilt from it | **none** — `recipe_ingredients_resolved`, `recipe_nutrition`, `recipe_tags` all SKIP |
| after removing the row | rebuilds clean, exit 0 |

The probe row is deleted in a `finally` and Gold is rebuilt at the end, so an interrupted proof
cannot leave poisoned data behind. Same reasoning as §13's mutation sweep: a gate nobody has
watched fail is not evidence of a gate.

**Distribution expectations (§4's second quality tool) — advisory, not a gate.** The dbt gate
asks "is any value impossible?" and blocks. `pantryiq.gold.expectations` asks a different
question — "has the shape moved?" — which bounds-checking cannot see, and reports rather than
refuses. Nine checks over published Gold: mean kcal/100g, median and **p99** total grams, mean
coverage, mean trust score, tagged share, conversion share, and hours since publish.

**Choosing the statistic mattered more than choosing the range, and I got it wrong first.** I
asserted these would have caught the pack-size bug, then checked. The **median does not move at
all** — 652.0 g before and after, because 218 of 15,000 recipes cannot shift a median. The tail
does: p99 goes 4,172 → 12,517 g, a clean 3.00×, which fires against the 7,000 bound. A suite of
means and medians would have passed straight through a 3× conversion error. The tail statistic is
in the list because it was measured to work, and the docstring records the false start.

### Orchestration (3.7)

`dags/pantryiq_pipeline.py` — 11 tasks, Bronze-derived Silver through the gated Gold build.
Every task calls the same `main()` a human would run by hand, so the DAG orchestrates the
documented interface rather than a second implementation that could drift from it.

**Strictly sequential, and that is correctness rather than style.** Silver and Gold share one
DuckDB file and DuckDB takes an exclusive write lock, so two concurrent writers do not
interleave — they fail. `max_active_tasks = 1`, asserted in a test so the next person to
"optimise" the DAG with parallelism finds out at test time.

**The gate is one task, not two.** The plan's original shape had `dbt_run` then `dbt_test` as
separate tasks; splitting them reintroduces exactly the hole 3.6 closes, so the DAG runs
`dbt build`.

**Ingestion is a separate, manual DAG.** The Bronze pulls are slow, and more importantly their
outputs are the frozen reference the 300 gold labels and every §7-§14 metric are pinned to.
Scheduling them would move the ground truth under the measurements without anyone deciding to.

Both DoD halves demonstrated:

| | result |
|---|---|
| end-to-end run | `state=success`, 11/11 tasks, dbt `PASS=25`, 15.2 s |
| re-run idempotent | identical row counts **and** an identical SHA of `gold.recipe_nutrition`'s contents |

The content hash matters more than the counts: a stale no-op — a step that silently did nothing
— would preserve every row count while leaving the data unchanged from a previous run. Only
hashing the values distinguishes "rebuilt to the same answer" from "did not rebuild".

Airflow lives in an isolated dependency group. With it: 404 tests. Without it (what CI runs):
399 pass, 1 skips. The isolation is verified rather than assumed.

## Still to measure
- **Dry mix vs ready-to-eat** as a rule-2b extension: `chocolate pudding`, `black cherry jello`
  and `coffee creamer` all fail this way, with 2–6× kcal consequences.
- ~~**A holdout read for the abstain threshold.**~~ **Done (2026-07-31, §12):** null recall
  58.3% on the holdout against 70.8% in-sample on tune, with CV at 62.5% between them.
  *(original entry: the holdout would confirm it, but that is a second read of the project's one
  clean estimate — worth spending only if abstention is to be a headline claim rather than a
  safety feature.)* It was spent as part of the §13 re-run, which re-read the holdout anyway to
  re-derive §7 after the tie-break fix.
- **The 9 three-way-split strings** are the most genuinely ambiguous rows in the dataset and are
  worth reading directly — they are where the task itself, not the pipeline, is under-determined.

**Done since this list was written:** 2.5 (§7) · the shipped resolver (§8) · the stratum
investigation, which found frequency was the wrong axis (§8) · the anchoring diagnostic (§7) ·
`recall_at_k` rule-6 duplicate credit (§2 — implemented, changes nothing) · deprioritization fix
measured and rejected (§8) · the multi-annotator ensemble (§9 — measured, not promoted) · the 85
low-confidence labels (§9 — measured; the ensemble could not improve them) · the 2.6 viability
test (§10 — the question turned out to be structurally unanswerable against these labels) ·
2.7, the entity map and the stratified corpus headline (§11) · occurrence-weighted equivalence
(§11) · a fitted no-match threshold, closing §11's stated limitation (§12).

---

## §15 — Phase 4: the agent, and what could be measured about it (2026-08-04)

Phase 4 puts a model on top of the warehouse. The claim being defended is that it **can only
speak from verified data**, and the only version of that claim worth making is one a machine can
check. So the phase's headline is not the answers — it is the guardrail's two rates.

### The guardrail, measured

> ⚠️ **RETRACTED — the numbers in this subsection were wrong, and wrong by construction.** The
> catch rate below was a **tautology**: the harness discarded an injection when `permitted()`
> accepted it, then counted it caught when `check()` — the same predicate — rejected it. Those
> are complementary halves of one function; they agreed on 90 of 90 injections and the statistic
> could only ever print 100%. An adversary designed to always defeat the guardrail scored 100%
> through it as well. On the honest denominator the rule below caught **40%**, because a ±10%
> tolerance around ~81 values covered **79% of the number line**. See **§16** for the rebuild and
> the corrected measurement. The original text is kept so the error is legible rather than tidied
> away.

`scripts/measure_guardrail.py`: 20 real questions through parse → retrieve → generate, then one
injected numeric error at a time.

| | ~~value~~ **retracted** | ~~95% Wilson~~ |
|---|---|---|
| ~~catch rate~~ | ~~36/36 = 100%~~ → **36/90 = 40%** | 30.5 – 50.3% |
| ~~false-positive rate~~ | ~~0/20 = 0%~~ (in-sample, after two fixes fitted on the same 20) | 0.0 – 16.1% |
| numeric claims checked | 346 (17.3 per answer) | |

Both numbers or neither. A guardrail that rejects everything catches 100% of fabrications and is
worthless; the unmodified answers passing **is** the control, which is the §13 lesson applied
before the fact rather than after it.

Five error classes were injected — a perturbed real figure, an invented cost, a fabricated
serving count, a total divided by its serving count, and a plausible absent round number. All
five classes were caught in full.

**54 of 90 injections (60%) were discarded** because the number landed on a legitimate value in
the same context, which makes them not fabrications. That rate is itself the finding: with 8
recipes the allowed set holds ~90 values and small integers are dense in it. **The guardrail
verifies a number EXISTS in the context, not that it belongs to the recipe being discussed** —
quoting one recipe's calories while naming another passes. That is misattribution, not
fabrication, and closing it needs per-recipe scoping this does not attempt.

### Two things the first measurement got wrong

1. **Recipe ids were being read as quantities.** Both initial "false positives" were a model
   citing `recipenlg:14797` — quoting the context exactly. Fixed by stripping ids that appear in
   the context before extraction; an invented id is still caught. FP rate 10% → 0%.
2. **The ordinal allowance left fabricated serving counts unguarded.** Small integers have to be
   tolerated as list markers, which meant "it makes 6 servings" passed. Now a unit following the
   number makes it a measurement, checked strictly; a bare integer stays an enumeration.

### Latency — the DoD is missed, and by how much

| stage | time |
|---|---|
| parse (Claude, effort low, thinking off) | 3.6 – 4.0 s |
| retrieval (SQL, no model) | **15 – 38 ms** |
| generation, first token | 1.3 – 2.3 s warm |
| **time to first token, median** | **5.3 s** (p90 8.2 s) |
| complete answer | 7 – 12 s |

The brief asks for **< 5 s end to end**, and that is not met. Both calls already run at the
minimum effort with thinking disabled, so there was no budget left to cut. Streaming was added
and moved the perceived wait from 13.4 s to a 5.3 s median first token; the remaining binding
constraint is the parse call, which must complete before retrieval can start. Sonnet 5 parses
identically in 2,354 ms against Opus 5's 3,646 ms and would close most of the gap — **not taken,
because model choice is the user's decision, not a silent optimisation.**

Retrieval's own budget was nearly missed for an avoidable reason: expressing the ingredient match
as a join against a table of patterns costs **973 ms** for three terms, because the pattern
becomes a column and DuckDB recompiles the regex for each of 112,463 rows. Bound as constants the
same three cost **17 ms** — a 57× difference in SQL that reads almost identically.

### The matching rule, chosen by measurement

| term | exact | substring | word-boundary | what the boundary kills |
|---|---|---|---|---|
| `corn` | 63 | 1,477 | **871** | `acorn`, `popcorn`, `mexicorn` |
| `milk` | 2,306 | 3,842 | **3,436** | `buttermilk`, `milky way bar` |
| `pepper` | 1,316 | 3,782 | **3,666** | `pepperoni`, `peppermint`, `cyapepper` |
| `chicken` | 211 | 1,565 | **1,565** | (nothing) |

Exact under-recalls catastrophically; substring reaches into different foods. The boundary rule
keeps genuine plurals and generalises — a hand-curated stoplist would never have caught
`cyapepper` or `mexicorn`.

### The dietary tags were wrong, and the fix is a different shape

Phase 3 stated `vegetarian` / `vegan` / `gluten-free` as **the absence of a disqualifying
substring in `display_name`**. A denylist cannot support a safety claim: every product name
nobody anticipated is a silent false positive. Phase 4's recipe titles gave the first independent
signal, and two failures fell straight out:

    recipenlg:9231  "Meat Loaf"  vegetarian   `hamburger`  -> "BURGER KING, Hamburger"
    recipenlg:10493 "Meat Loaf"  gluten-free  `quaker oat` -> "Cereals, QUAKER, MultiGrain Oatmeal"

Neither string contains a listed token. **My Phase-3 leak test reported 0 leaks because it used
the same kind of substring pattern as the predicate it was testing** — the test and the rule
shared a blind spot, so it confirmed the denylist against itself.

The fix pulls USDA's own `foodCategory` (a third Bronze table, ~410 requests — the portions cache
had kept only `fdcId` and `foodPortions`) and inverts the logic: **a tag now requires every
ingredient to sit in an allowed one of USDA's 25 food groups.** An unrecognised food declines the
tag instead of qualifying for it. Name patterns are kept as a *second* filter, because groups are
coarse and some of them mix — "Fats and Oils" holds lard, "Soups, Sauces, and Gravies" holds beef
broth.

| tag | denylist | + allowlist | + directions veto | title leak |
|---|---|---|---|---|
| vegetarian | 604 | 580 | **572** | 13 → **5** |
| vegan | 147 | 139 | **134** | 9 → **4** |
| gluten-free | 329 | 68 | **63** | ? → **1** |

> ⚠️ **The "13 → 5 / 9 → 4" figures are wrong twice over.** They are *title-leak* counts, not
> false-tag counts, and they were produced by a regex that exists nowhere in the repo — the one
> pinned in `tests/test_recipe_tags.py` measures 8 → 1, so `assert leaked <= 5` had five times
> the slack it appeared to. Measured properly, by inspecting the resolved entities, the group
> allowlist left **30 of 572 vegetarian (5.2%) and 16 of 134 vegan (11.9%)** falsely tagged — six
> and four times what was published. Two recipes tagged **vegan** were a ribs recipe and a
> cocktail-wieners recipe. See **§16**.

### `coverage = 1.0` does not mean the ingredient list is complete

The residual leaks turned out not to be tag failures at all. **The corpus ships truncated
ingredient lists.** Bronze's "Pickled Bologna" lists vinegar, sugar, salt and pickling spice —
no bologna. Silver kept all four lines faithfully; the pipeline is correct end to end. So a
recipe can be fully weighed, fully resolved, sit entirely inside allowed food groups, and still
not be vegan.

`nutrition_coverage = 1.0` means *every line we have was weighed*, not *we have every line*, and
nothing downstream of the ingredient list can see the difference. The **directions** can: they
say "remove skin from 2 rings bologna". A directions-based veto now blocks a tag when the method
names a food the ingredient list does not.

**The measurement is deliberately independent of the fix**: the veto reads directions, and the
residual leak above is measured with *titles*. Of the 5 vegetarian titles still naming meat, at
least three are artifacts of the measurement rather than the tags — "Cucumber Sauce **For**
Fish", "Fish Fry Coating Mix" and "Meat Marinade" are condiments that contain no meat.

**And the first version of that measurement was vacuous.** It matched 0 of 15,000 recipes, which
looked like a clean result. `directions` is stored as a JSON *string*, so joining it as a list
spaced out every character (`'b o i l   i n g r e d i e n t s'`) and the pattern could never fire.
Caught by running the control: the corrected pattern fires on 3,083 of 15,000 (20.6%). *A check
that cannot fail is not evidence.* Same lesson as §13's mutation sweep, in a new costume.

### Design decisions Phase 4 made and had to defend

- **`kcal_basis` is parsed, not assumed.** "Under 500 calories" means a serving to almost
  everyone. The median recipe here is 2,539 kcal in total, so applying that bound to the total
  returns 145 recipes that are mostly dips and dressings, while per-serving returns 104 actual
  meals. Getting the basis wrong answers a meal question with salad dressing.
- **Division is not a permitted derivation.** A recipe can carry `servings: 8` and
  `kcal_per_serving: unknown` at once, because §3.4 refused to publish the quotient below full
  coverage. 3,470 / 8 = 433.75 is exactly the figure that refusal exists to prevent.
- **Sums are not permitted either.** Allowing subset sums over 8 recipes admits 256 extra values
  per field. Measured first: the false-positive rate without them is 0%, so they were not bought.
- **The planner runs on totals and says so.** Only 118 recipes carry a per-serving figure and
  just 14 of those also have complete cost. A per-day-per-person plan would rest on a denominator
  invented for the rest.
- **Verified totals need their own channel.** Generation is forbidden from summing — and so it
  refused to describe its own week plan, saying it could not total costs across days. An
  `<already_computed>` block now carries figures the planner calculated *and verified*, which are
  facts rather than arithmetic the model performed.

### 4.8 — the quality explainer, and two more identifier bugs

Design doc §97, and part of the Phase-4 gate at §124 (which I had earlier described as excluding
it — it does not). Two triggers, both from artifacts that already exist: dbt's
`target/run_results.json`, and the 2,884 of 9,163 strings (31.5%) the resolver abstained on or
flagged. Low-confidence findings are ranked by **occurrence, not cosine** — a string the resolver
was unsure about that touches 1,811 recipe lines outranks one touching 1, because operator
attention is the scarce resource the table exists to direct.

**The explainer is guardrailed like a user-facing answer.** An ops summary that invents a row
count is worse than no summary: it will be believed and acted on by someone without the query in
front of them. A rejected draft degrades to the facts themselves rather than being published.

**Watched failing, per the same standard as the gate proof.** `prove_gate.py`'s impossible row was
injected, `dbt build` run against staging, and the resulting `run_results.json` captured as a
fixture — a real artifact, not a hand-written one. It records **1 failing row and 17 downstream
models skipped**, and that propagation count is what tells an operator nothing reached Gold.

The first run of it against that real failure was **rejected by the guardrail**, and both causes
were mine:

1. `kcal_per_100g` — the *column name* that failed. Its digits were read as a fabricated "100".
   Same class as the recipe-id false positive in §15 above, and fixed the same way:
   `AnswerContext.identifiers` now carries names whose digits are not quantities, stripped
   longest-first so a shorter name that is a prefix of another cannot leave digits behind.
2. `910` — the test's configured upper bound. The model wrote "outside the configured bounds
   (0 to 910)", which is correct and was not in the ledger. Now read from `manifest.json`'s
   `test_metadata.kwargs`, rather than parsed out of the mangled test name
   (`..._kcal_per_100g__910__0.b67c7d28cb`) which is a string, not a data structure.

With both fixed the same failure produces a summary that passes on 7 numeric claims and correctly
infers, from the write-audit-publish design, that downstream consumers are reading *stale* data
rather than wrong data. **Percentages are precomputed into the ledger** for the same reason as the
planner's totals: the model will write "2,884 of 9,163 (31.5%)" correctly, and a guardrail holding
only the two counts would reject a true sentence.

---

## §16 — The Phase-4 review, and rebuilding what it broke (2026-08-04)

Five reviewers were fanned out over the finished phase. They found that **both of Phase 4's
headline claims were false**, and in both cases the error was structural rather than arithmetic:
a statistic that could only report one value, and a rule whose failures were unbounded.

### 16.1 The catch rate was a tautology

`measure_guardrail.py:214` discarded an injection when `permitted(injected, …)` returned true;
line 218 counted it caught when `check()` failed — and `check` decides by calling that same
`permitted` against the same allowed set. Discard and catch were complementary halves of one
predicate.

```
cases where (caught) == (not permitted(injected)):  90/90
disagreements — i.e. independent information:        0
```

Run through that harness, an adversary that *always* defeats the guardrail also reports 100%.
The only thing that could ever drive it below 100% was a failure of the number *extractor*.

**The fix is not a better discard rule, it is no discard rule.** An injected number is a
fabrication because the script put it there; whether it collides with some other real value
explains *why* the guardrail misses and is now reported beside the rate, not subtracted from it.

### 16.2 Why the old rule caught 40%

`error(v, c, floor=0) < 0.10` is symmetric on the max, so each allowed value claimed a **21%-wide**
interval rather than 10%. Over a median 81 values per context that covered:

| measure | median | range over 20 contexts |
|---|---|---|
| linear coverage of [20, 3000] | **78.9%** | 38.4 – 95.4% |
| P(a fabricated calorie figure is accepted) | **78.0%** | 38.1 – 95.6% |

And the tolerance bought nothing: of 346 numeric claims in 20 real answers, **345 were exact
matches and 1 was a list marker. Zero needed the band.** It came from `er/nutrition.EQUIVALENT`,
where 0.10 means "two USDA foods are nutritionally interchangeable" — a category error when
reused as quoting accuracy.

### 16.3 The rebuilt guardrail

Three changes, each closing a measured failure:

1. **Scope is per sentence, not the whole context.** A number is checked against the recipe the
   sentence is about. Misattribution — every figure real, attached to the wrong dish — was
   measured at a median **133% error, up to 1,105%**, and in 7 of 20 contexts it also upgraded an
   incomplete recipe to "6 of 6 ingredients weighed", falsifying the coverage guarantee. A
   sentence naming two recipes puts both in scope, because "X and Y, at 1,947 and 2,731 kcal
   respectively" is a correct sentence that section-level scoping rejected.
2. **Counts are separated from measurements by kind, not magnitude.** `EXACT_BELOW = 20` treated
   every small number as a count and rejected "about 13 g of protein" against a stored 13.1 —
   **51% of rounded sub-20g macro quotations**. The old 0% false-positive rate was clean only
   because those 20 questions quote `total_kcal`, which is always ≥ 20.
3. **List markers are positional, not vocabulary.** The old rule allowed any small integer unless
   a word from a fixed unit list followed it — the same denylist anti-pattern this project
   condemns in the dietary tags — and let through "It serves 2", "Makes 2 portions", "$2 total".

| | before | after |
|---|---|---|
| **catch rate, all injections** | 36/90 = **40.0%** | **147/170 = 86.5%** (Wilson 80.5–90.8%) |
| misattribution | 0/20 caught | **20/20** |
| division by servings | 1/10 | **10/10** |
| false positives | 2/20 (in-sample 0/20 after fitting) | **1/20 flagged, 0 genuine** |
| acceptance coverage | 79% of the number line | 1% band, per-sentence scope |

**The single flagged answer is a true positive.** The model wrote *"Cost is at least $2.22 —
sorry, at least $3.65"*, fabricating a figure and correcting itself mid-sentence; the guardrail
caught the wrong one. This is the project's **first recorded true positive on natural model
output** — every previous firing had been a false positive.

Per class, the weakest are now `whole-dollar invented cost` (10/20) and `serving count, reworded`
(14/20). Both fail for the same reason: small integers are dense in the allowed set — 38% of all
injections collide with some real value — and `costed_ingredients` had to be added to the ledger
because answers legitimately say "3 of 5 ingredients priced".

Four bugs were found *by* the rebuilt rule and fixed: `"Potato Casserole"` matching inside
`"Hash Brown Potato Casserole"` and splitting one recipe's section in two; nested parentheticals
(`Honey Oatmeal Drop Cookies(Makes 22 (2-Inch) Cookies)`) never matching; digits inside a title
being read as claims; and identifier stripping running *after* range-splitting.

### 16.4 The dietary tags, rebuilt on per-entity classification

The group allowlist was the second proxy to fail. Measured by entity inspection rather than by
titles, it left **30/572 vegetarian (5.2%) and 16/134 vegan (11.9%)** falsely tagged. Causes:
Worcestershire sauce (anchovies — 12 vegetarian, 6 vegan), marshmallows (gelatin — 17 and 4), and
a directions veto ending `)\b` instead of `)\w*\b`, so **no inflected form matched at all** —
`hamburger`, `sausages`, `steaks`, `chickens` all missed, while the gluten pattern one line below
had the suffix.

**A patch list fixes the four foods on it.** So the question is now asked directly, once per
entity: `er/dietary.py` classifies all 8,187 USDA entities as vegetarian / vegan / gluten-free /
unknown via Claude, into `silver.entity_dietary_flags`, and a tag requires **every** ingredient to
be `yes`. A food nobody has looked at is classified the same way as one that has.

Definitions are pinned in the module rather than left to the model (cheese counts as vegetarian
unless the description names animal rennet; oats are `unknown` for gluten because USDA does not
record certification). 705 of 8,187 entities come back `unknown`, and unknown disqualifies.

**The hand-written regression set was wrong and the classifier corrected it.** Four of thirty
verdicts in the first draft were mistakes, *all in the unsafe direction* — it claimed
Worcestershire sauce is gluten-free (barley malt vinegar) and russian dressing vegetarian (it
commonly contains Worcestershire, hence anchovy). The set now records acceptable *ranges* and
asserts the property that actually matters unconditionally: **never a wrong `yes`** (0 of 30).

| tag | Phase-3 denylist | group allowlist | **entity classifier** |
|---|---|---|---|
| vegetarian | 604 | 572 | **418** |
| vegan | 147 | 134 | **61** |
| gluten-free | 329 | 63 | **179** |

Gluten-free *rose* because the group allowlist was too coarse in both directions — it excluded
whole categories containing gluten-free foods while admitting meatless analogues, which are
filed under Legumes and made of wheat gluten.

Measured against signals the rule does not read:

- **0** tagged recipes contain an entity in a USDA flesh food group
- **0** tagged recipes contain an entity classified anything but `yes`
- **0** gluten-free recipes whose own ingredient lines name unqualified grain
- title leak: **3/418 vegetarian, 3/61 vegan** — and all three are artifacts of the measurement:
  "Cucumber Sauce **For** Fish", "Cold Pack Fish" (vinegar and salt) and "Fish Fry Coating Mix"
  (cornmeal and spices) contain no fish

Three filters are kept alongside the classifier because each catches what the others cannot: the
coverage gate, a name denylist (the classifier judges the *entity*, and entity resolution is
67.3% accurate — three recipes containing real chicken, bacon and sausage resolved to the
*meatless* analogues), and the directions veto for truncated ingredient lists.

**A fourth check was dead on arrival and caught by running it.** `has_gluten_line` used
`SIMILAR TO '%(flour|...)%'`, and DuckDB's `SIMILAR TO` is anchored regex in which `%` is a
*literal percent sign* — it matched nothing. Rewritten with `regexp_matches` and verified against
real strings before being trusted. That is the third time in this project a check has been found
that could not fire.

### 16.5 The safety assertions now run inside the gate

They previously lived only in pytest, which meant they ran on a developer machine against a
pre-built export and could block nothing — three mutations to `recipe_tags.sql`, including adding
`'Beef Products'` to the vegetarian allowlist, left all 520 Python tests green because `make test`
never invokes dbt. Three singular tests now run under `dbt build --target staging`.

**Watched blocking, not asserted:** with the safeguards removed, they failed on 371 and 83 rows,
`dbt build` errored, and `publish()` never swapped. Restored, all 32 dbt tests pass.

### 16.6 The test suite

A mutation sweep with a proper control — which also **reproduced the previous sweep's bug**: the
old script copied only `src tests scripts docs pyproject.toml`, omitting `seeds/`, so every run
died in `test_cost_reference.py` before reaching a Phase-4 test and reported `CAUGHT`
unconditionally.

**62 mutations, 32 survived (48%).** The most serious: `guarded_answer` and `explain.run()` had
**no tests at all** — inverting either left all 520 green, and `guarded_answer` was genuinely
wrong, returning the templated fallback paired with the *rejected retry's* verdict, so the query
log recorded a clean answer alongside `guardrail_pass=False` and numbers appearing nowhere in it.
Four tests named a property they did not check, including a trust-tiebreak test that passed with
trust deleted from the `ORDER BY`.

Suite is now **548 tests**, up from 520.

### 16.7 Latency, unchanged and restated

Opus 5 is retained on both calls by decision. Time to first token is a **median 5.3 s** (p90
8.2 s) against a <5 s target; retrieval is 15–38 ms and the binding constraint is the parse call,
which must complete before retrieval can start. Sonnet 5 parses identically in 2,354 ms against
Opus 5's 3,646 ms and would close the gap — not taken, because model choice is the operator's
decision rather than a silent optimisation.
