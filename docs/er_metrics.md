# Entity resolution — measured results

Regenerate: `uv run python -m pantryiq.er.metrics` (recall@k) ·
`uv run python -m pantryiq.er.nutrition` (kcal error + ambiguity ceiling) ·
`uv run python -m pantryiq.er.convention` (rule 2b audit) ·
`uv run python -m pantryiq.er.report` (2.5 holdout) ·
`uv run python -m pantryiq.er.resolve` (shipped resolver) ·
`uv run python -m pantryiq.er.relabel --report --pass 1` ·
`uv run python -m pantryiq.er.adjudicate --report` ·
`uv run python -m pantryiq.er.ensemble --report` (annotator agreement) ·
`uv run python -m pantryiq.er.adjudicator_value --report` (2.6 viability)

**If you read one thing:** the headline metric of this project is **nutrition error, not entity
accuracy**, and the reason is measured rather than asserted — see
[The ambiguity ceiling](#the-ambiguity-ceiling). A median ~32% kcal spread exists *within* the
correct food, so no entity-level precision figure can carry the weight the plan originally
assigned it.

**The 2.5 result, on the frozen holdout: 63.6% of resolved strings are nutritionally equivalent
to the gold label (95% CI 53.2–74.0%), median kcal error 0.0%.** Eight engineered features and a
calibrated logistic model are worth nothing measurable over taking the top embedding cosine — see
[§7](#7-scoring-and-routing-25--the-frozen-holdout-read-once).

**Label quality, measured: Krippendorff's α = 0.709** across three independent annotators on 120
strings ([§9](#9-multi-annotator-ensemble--α--0709-and-it-does-not-beat-the-gold-labels)). Usable
for directional claims, not precise ones — and the ensemble could not improve on the existing
labels, so they were left in place.

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
**Krippendorff's α = 0.709** (§9) — below the ~0.8 publication bar, above the ~0.67 floor. Quote
that alongside any entity-level figure: it is the measured version of everything above.

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
| **nutritionally equivalent (<10% kcal)** | **49/77 = 63.6%** (95% CI 53.2–74.0%) |
| median relative kcal error | **0.0%** (CI 0.0–6.8%) |
| mean relative kcal error | 19.4% (CI 12.9–26.4%) |
| entity top-1, of strings whose gold was retrieved | 46.8% |
| entity top-1, of all 99 holdout strings | 37.4% |
| retrieval ceiling on this split | 90.8% |

| error band | n |
|---|---|
| within 10% (nutritionally equivalent) | 49 |
| 10–25% | 6 |
| 25–50% | 9 |
| over 50% (materially wrong) | 13 |

By stratum: head 66.7% (CI 48.1–81.5), **mid 50.0%** (32.1–67.9), tail 77.3% (59.1–95.5). Mid is
worst, the same inversion recall@k shows.

**Read against the ceiling, not against 100%:** the ambiguity ceiling on these same strings is a
median **24.3%** kcal spread *within* the correct food.

### The learned scorer does not beat top-cosine

| split | difference in kcal-equivalence (model − baseline) | McNemar |
|---|---|---|
| tune (out-of-fold) | **+3.8pp** [−0.6, +8.2] | 4 / 10, p=0.18 |
| **holdout** | **−3.9pp** [−13.0, +3.9] | 7 / 4, p=0.549 |

**The direction flipped out of sample** — which is what a noise difference does. The paired test
at step 3b said "cannot distinguish"; the adoption decision over-read a consistent point-estimate
direction as weak evidence, and the holdout corrected it. Eight engineered features, a logistic
model and isotonic calibration are worth *nothing measurable* over taking the top embedding
cosine.

Switching to the baseline *because the holdout prefers it* would be selection on the holdout and
would void the 63.6% estimate. So that figure stands as the committed pipeline's number. The
defensible argument for shipping the simpler resolver is that the two are indistinguishable on
**both** splits, so simplicity breaks the tie.

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

### Anchoring diagnostic

| presentation | n | kcal-equivalent |
|---|---|---|
| control (unranked, no suggestion) | 7 | 57.1% |
| ranked | 78 | 57.7% |

A clean null — no evidence the gold labels encode the ranking they were shown. n=7, so a smoke
test rather than a validation.

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
  anchoring §7 spent a control group measuring. `shown_order` is recorded on every judgment.
- **The rules are sliced out of `docs/labeling_guide.md` at runtime**, not restated in the prompt.
  Restating them is precisely the drift that produced §3's 36% disagreement.
- **Recipe lines are treated as data**, wrapped in a tag with an explicit instruction to ignore
  directives inside them (the brief's injection-hardening requirement).

### The agreement number

**Krippendorff's α (nominal) = 0.709** over all 120 strings; **0.780** over the 25
mixed-confidence validation strings. Unanimous on **74/120 (61.7%)**. **9 strings** were three-way
splits — three capable models, three different answers, no majority at all.

0.709 sits below the ~0.8 publication bar and above the ~0.67 floor. **It confirms the §1 caveat
rather than dissolving it:** these labels support directional conclusions, not precise ones. That
was already the doc's prose position; it is now a statistic a reader can interpret.

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
| shipped resolver — top cosine (§7 holdout) | 46.8% | 63.6% |
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

## Still to measure

- **2.7** — assemble `silver.ingredient_entity_map` over all 9,324 strings and publish the
  headline. This is the deliverable that makes the work demonstrable rather than a set of
  measurements.
- **Occurrence-weighted equivalence.** Head strings carry 83.5% of all ingredient occurrences and
  score better than mid, so the per-occurrence figure will differ substantially from 63.6% —
  probably upward. The brief asks for both denominators, and the gold set is a stratified sample
  with known stratum sizes, so the correct treatment is a stratified estimate rather than a raw
  average.
- **Dry mix vs ready-to-eat** as a rule-2b extension: `chocolate pudding`, `black cherry jello`
  and `coffee creamer` all fail this way, with 2–6× kcal consequences.
- **The 9 three-way-split strings** are the most genuinely ambiguous rows in the dataset and are
  worth reading directly — they are where the task itself, not the pipeline, is under-determined.

**Done since this list was written:** 2.5 (§7) · the shipped resolver (§8) · the stratum
investigation, which found frequency was the wrong axis (§8) · the anchoring diagnostic (§7) ·
`recall_at_k` rule-6 duplicate credit (§2 — implemented, changes nothing) · deprioritization fix
measured and rejected (§8) · the multi-annotator ensemble (§9 — measured, not promoted) · the 85
low-confidence labels (§9 — measured; the ensemble could not improve them) · the 2.6 viability
test (§10 — the question turned out to be structurally unanswerable against these labels).
