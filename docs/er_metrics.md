# Entity resolution — measured results

Regenerate with `uv run python -m pantryiq.er.metrics`.

## ⚠️ Label provenance — read before quoting any number

The 300 gold labels are **not all human-produced**:

| Labeler | Count | What it means |
|---|---|---|
| human | 18 | Independent ground truth |
| **claude (LLM annotator)** | **282** | Model-produced, following `docs/labeling_guide.md` |

This was a deliberate trade: labeling 300 strings by hand costs 8–12 hours, and the project
owner chose LLM annotation over abandoning the metric. The consequence must be stated wherever
these numbers appear:

- **This measures agreement between the local pipeline and an LLM annotator, not accuracy
  against human ground truth.** LLM-as-annotator is an accepted practice, but it is weaker
  evidence, and the headline claim must say so.
- **The 2.6 Claude adjudicator cannot be honestly evaluated against these labels.** Scoring a
  Claude adjudicator against Claude-written labels measures self-consistency, not correctness.
  Adjudicator accuracy must be reported on the human-labeled subset only, or not claimed.
- **recall@k on the Claude subset is biased upward.** The annotator saw the generated candidate
  list before deciding, so its labels lean toward entities the generator surfaces. The
  human-labeled subset is the unbiased check — see the provenance split below.

Every label records its `labeler` and a `confidence` flag (215 high / 85 low). The 85
low-confidence labels are the highest-value targets for human review.

**To strengthen this:** a human relabeling ~30 randomly drawn strings — without seeing the
existing labels — would give a measured human/LLM agreement rate, converting the caveat above
from a qualitative disclaimer into a number.

## Candidate generation (2.4)

8,187 USDA entities → ~78 candidates per ingredient string, by exact chunked cosine (no ANN)
plus head-noun token blocking. `recall@k` is the **ceiling on everything downstream**: a
scorer cannot pick an entity that was never shown to it.

| k | recall |
|---|---|
| 5 | 68.6% |
| 10 | 75.0% |
| 25 | 84.8% |
| **50 (generation k)** | **90.5%** |
| 100 | 92.0% |

Null class excluded from the denominator: 36 of 300 strings (12.0%) have no reasonable USDA
match and are labeled `no-match`. They are a real answer, not a retrieval failure —
`bisquick`, `kitchen bouquet`, `old bay seasoning`, `orange kool-aid` genuinely are not in
Foundation + SR Legacy.

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
two text forms were introduced — the embedding is fairly robust to facet order. Token blocking
alone is weak but cheap, and it is a genuine complement: it retrieves by exact head-noun match
where embeddings drift semantically (`soda` → soft drinks rather than baking soda).

> **Measurement bug found and fixed here.** The first version recorded a single `method` per
> candidate — whichever generator found it *first*. Since `embed_search` was consulted first it
> claimed nearly every row, making the other two look useless (`embed_desc` 1.1%, `token_head`
> 0.4%) and the A/B unmeasurable. Membership is now recorded per generator as three independent
> booleans, and the generators overlap heavily by design.

### By label provenance

| Labeler | n | @5 | @25 | @50 |
|---|---|---|---|---|
| human | 17 | 88.2% | 94.1% | 94.1% |
| claude | 247 | 67.2% | 84.2% | 90.3% |

The human subset scores *higher*, which is reassuring — had the LLM labels been simply
rubber-stamping the top candidate, the LLM subset would score near 100%. It does not. But n=17
is far too small to draw a firm conclusion; treat this as a smoke test, not a validation.

## Still to measure

Precision / recall / F1 (2.5, on the frozen holdout with bootstrap CIs), % resolved without an
LLM call on both denominators, and adjudicator accuracy (2.6) — which, per the provenance
warning above, is only meaningful on human-labeled rows.
