# Gold labeling guide — ingredient string → USDA entity

This document defines what a **correct** match is. It exists because the hard part of this
problem is not string similarity but **specificity**: `"1 tsp. vanilla"` has 89 plausible USDA
candidates, `"2 eggs"` has 110, `"1/2 c. butter"` has 163, and there are 296 lamb entries. Two
labelers — or the same labeler on two different days — will disagree on those unless the rule
is written down first. Precision/recall computed against inconsistent labels is noise.

**These labels are the ground truth for the headline metric.** The model is measured against
your judgment, never the other way around.

> ⚠️ **Provenance (2026-07-24):** 282 of the 300 labels were ultimately produced by an LLM
> annotator following this guide, not by a human — hand-labeling proved to cost 8–12 hours.
> Every label records its `labeler`. This materially weakens the headline claim and changes
> what can be evaluated; see `docs/er_metrics.md` before quoting any metric.

---

## The rules

### 1. Prefer the least-qualified base form

Pick the most generic entry that is consistent with the ingredient line. USDA descriptions are
comma-inverted facet strings, and each extra facet is an extra claim the recipe did not make.

| Line | Correct | Not |
|---|---|---|
| `2 eggs` | `Egg, whole, raw, fresh` | `Egg, whole, cooked, hard-boiled` |
| `1/2 c. butter` | `Butter, salted` | `Butter, whipped, with salt` |
| `1 lb. ground beef` | `Beef, ground, unspecified fat content, raw` | `Beef, ground, 80% lean meat / 20% fat, patty, cooked, broiled` |

### 2. Go specific only when the line does

Every qualifier in the line should be honoured; qualifiers *not* in the line should not be
invented.

| Line | Correct |
|---|---|
| `1/2 c. unsalted butter` | `Butter, without salt` |
| `1 c. whole milk` | `Milk, whole, 3.25% milkfat` |
| `1 c. milk` | `Milk, reduced fat, fluid, 2% milkfat` only if nothing more generic exists — otherwise the unqualified entry |

Preparation words that survived normalization (`cooked`, `dried`, `frozen`, `ground`) are
deliberate: they change the food, so honour them.

### 3. Skip babyfood, restaurant, and branded entries unless the line names them

`Babyfood, …`, `Fast foods, …`, `APPLEBEE'S, …`, `GERBER …` are real USDA rows but almost never
what a recipe means. `silver.usda_foods.is_deprioritized` flags them (14.9% of the corpus).
Choose one only if the line explicitly names the brand or restaurant.

### 4. When nothing fits, say `no-match` — do not force it

The null class is a real answer and roughly a third of the tail deserves it. USDA
Foundation + SR Legacy has no entry for `jell-o cherry flavor gelatin`,
`chocolate fudge cake mix without pudding`, or `7-up`.

Say `no-match` when no entry is a reasonable **nutritional** stand-in. Do not stretch to a
distant relative — a wrong match is worse than an honest gap, because a wrong match silently
produces wrong nutrition downstream.

### 5. Tie-breaker: prefer the entry that has nutrition data

When two entries are equally valid under rules 1–2, choose the one with a non-null
`kcal_per_100g`. This is not cosmetic: the Foundation entries `Butter, stick, salted` and
`Butter, stick, unsalted` carry **no energy value at all**, while SR Legacy `Butter, salted`
has 717 kcal/100g — and butter is the 5th most common ingredient in the corpus. 73 foods
(0.9%) have no energy value.

### 6. Duplicates: either id is correct

94 descriptions appear **twice**, once as Foundation and once as SR Legacy, with different
`fdc_id`s and slightly different values (`Carrots, baby, raw` → 38.3 vs 35.0 kcal). Pick
either; under rule 5 prefer the one with nutrition.

> Evaluation credits **any entity whose `description_raw` matches the gold label's**, so a
> prediction is not penalized for choosing the other member of a duplicate pair. Scoring on
> raw `fdc_id` equality alone would understate precision.

---

## What gets labeled

300 distinct normalized strings, sampled stratified by frequency and **seeded** so the sample
is reproducible:

| Stratum | Definition | Distinct in corpus | Sampled |
|---|---|---|---|
| head | ≥ 20 occurrences | 545 | 100 |
| mid | 3–19 occurrences | 1,580 | 100 |
| tail | 1–2 occurrences | 7,228 | 100 |

> **Reduced from 500 on 2026-07-24.** 500 labels proved to cost 8–12 hours of human judgment,
> not the ~90 minutes originally estimated. The reduced set is a *prefix* of the original
> per-stratum draw rather than a fresh draw — the original was already shuffled within each
> stratum, so a prefix is still a random subsample, and it preserves labels already collected.
> The cost is wider confidence intervals (~±10pp rather than ~±6pp), reported as such.

Stratifying matters because the corpus is extremely skewed: 545 head strings account for
**83.5% of all 112,463 ingredient occurrences**. Sampling by occurrence would produce a set of
almost entirely easy strings (`sugar`, `salt`, `egg`) and flatter the metric. Sampling by
distinct string over-weights the hard tail. Reporting both — unweighted and
occurrence-weighted — is the honest answer, and requires the strata.

The 300 are split **201 tune / 99 frozen holdout**, assigned at sampling time so the split
cannot be influenced by what the labels turn out to be. The splits are *interleaved* (every
third row) rather than assigned by position, so stopping early still yields both in
proportion rather than all tune and no holdout. Thresholds and the calibration model
are fitted on the tune split only; every reported precision/recall/F1 number comes from the
holdout. The holdout's fingerprint is recorded in `data/gold_labels/manifest.json` —
if it changes, the reported metrics are void.

## The anchoring control

The CLI ranks candidates and lets `ENTER` accept the top one. That is a real speed-up and a
real risk: an accept rate cannot be distinguished from genuine agreement if every row is
presented the same way.

So **~15% of rows (42 of 300) are shown scrambled and unranked**, with no suggestion and no
`ENTER` shortcut. Comparing how often the labeler's choice matches the model's top pick in
each mode gives a measured estimate of how much the ranking anchored the judgment. That number
gets reported alongside precision/recall rather than assumed away. Control assignment is
deterministic per string, so it is stable across re-runs.

## Labeling honestly

- **Label the string, not the model's suggestion.** The CLI shows 5 candidates (`m` expands to
  25), but it can be wrong; `s` searches all 8,187 foods. If the right answer is not in the
  list, find it — that is precisely the signal the recall@k ceiling measures. Accepting a
  wrong top row because it is convenient corrupts the metric in the model's favour.
- **"Nothing similar in USDA" is a hint, not a verdict.** When the best candidate is weak the
  CLI says so, because much of the tail genuinely has no match. It is still your call.
- **When genuinely torn, prefer the more generic entry** (rule 1) and move on. Consistency
  matters more than any single call.
