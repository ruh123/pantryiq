# Phase 2 — Silver + Entity Resolution (plan)

The headline differentiator. Runs on the 15K subset; entity resolution operates on
**distinct normalized ingredient strings** (dedup + cache) so it stays cheap/fast.
Ordered increments, each with a definition of done. The earlier design-review P0s are
folded into the relevant steps.

| # | Increment | Definition of done | Addresses (review) |
|---|---|---|---|
| 2.1 | **Parse recipe ingredient lines** — `bronze.raw_recipes` → `silver.recipe_ingredient_lines` (quantity, unit, ingredient_text); normalize (lowercase, strip filler, singularize); emit DISTINCT normalized strings | parser tested on tricky lines ("2 (16 oz.) pkg…", "juice of 1 lemon", "a pinch of…"); % parsed; distinct-string count | — |
| 2.2 | **Normalize USDA foods** — `bronze.raw_usda_foods` → `silver.usda_foods` (canonical_name, category, nutrients/100g); map energy-name variants (`Energy` / `Energy (Atwater General/Specific Factors)` → one kcal) | kcal present ~100%; sanity bounds pass | energy-name P0 |
| 2.3 | **Gold labeled set** — sample distinct strings STRATIFIED by frequency (head/mid/tail); hand-label correct USDA `canonical_id` OR explicit "no-match"; from raw lines not NER; ~500 labels + a FROZEN holdout slice; build a labeling CLI | ~500 labels w/ strata + null class; **human judgments = ground truth** | gold stratification, null class, tune/report split |
| 2.4 | **Blocking with measured recall** — candidate USDA foods via embedding-ANN (all-MiniLM-L6-v2 + hnsw) ∪ token blocking | recall@k curve vs gold; publish the recall ceiling | blocking-recall P0 |
| 2.5 | **Scoring + calibrated routing** — features (Jaro-Winkler/token-set + embedding cosine) → calibrated match model (logistic/isotonic) fit on gold TRAIN; auto/adjudicate/flag thresholds at a precision target; report P/R/F1 on frozen HOLDOUT w/ CIs | P/R/F1 (overall + per-stratum) w/ CIs; auto/adjudicate/flag split | calibration + tune/report P0 |
| 2.6 | **Claude adjudication** (mid-confidence) — raw text + top-3 → pick or "no-match"; recipe text treated as data (injection-hardened), temp 0; cache by (normalized string, candidate-id set) | adjudicator accuracy on mid-confidence slice; % resolved w/o LLM (per-occurrence AND per-unique-string) | adjudicator injection/cache/self-eval P0 |
| 2.7 | **Assemble `silver.ingredient_entity_map`** + headline metrics | entity map populated; P/R/F1 (overall/per-method/per-stratum), recall ceiling, %-no-LLM — the README headline | — |

**Human inputs needed (not at the start):**
- `ANTHROPIC_API_KEY` in `.env` — only at 2.6.
- Hand-labeling ~500 gold rows at 2.3 (a labeling CLI will speed it up; the judgments are the ground truth).

**Deferred to Phase 3** (belong with nutrition/Gold, not ER): unit→gram conversion + servings,
the write-audit-publish quality gate, cost table, dietary tags — each a first-class tested component.

**Suggested start:** 2.1 (parse recipe ingredient lines) — foundational, no new keys, builds on the
existing `QTY` work in `scripts/inspect_recipenlg.py`.

**Open question from the (incomplete) ER-plan review:** confirm whether the abridged FDC payload
carries `foodCategory` — 2.4 token/category blocking and `silver.usda_foods.category` depend on it.
The ER-plan reviewer did not finish (session limit); re-run it next session for a full ML critique.
