# PantryIQ

A verified recipe data platform with a data-grounded AI layer. The headline is **entity
resolution**: free-text recipe ingredients resolved to canonical USDA food entities with
measured precision/recall — so the AI layer can only speak from verified nutrition, cost, and
ingredient data, with a deterministic check on every numeric claim it makes.

- **Operating brief (locked decisions, phase gates):** [`docs/PantryIQ_Master_Prompt.md`](docs/PantryIQ_Master_Prompt.md)
- **Full design:** [`docs/PantryIQ_Design_Doc.md`](docs/PantryIQ_Design_Doc.md)

## Results

**All 9,324 distinct ingredient strings are resolved** to USDA entities with nutrition attached,
in `silver.ingredient_entity_map`, covering 112,452 of the corpus's 112,463 ingredient lines.

| Headline (stratified over the corpus) | per unique string | per occurrence |
|---|---|---|
| **nutrition within 10% of the gold label** | 62.0% [53.6, 70.1] | **70.1%** [51.3, 81.6] |
| entity top-1 | 40.7% [32.5, 49.0] | 51.6% [27.3, 68.8] |

The per-occurrence column is the one that describes the product: head strings are 5.8% of the
vocabulary but 83.5% of what a user actually hits. Both figures are estimates of agreement with
the gold labels — see the caveat below.

Three results that shaped the build, each measured rather than assumed:

- **There is a ceiling this pipeline cannot pass, and it is in the data, not the model.** The
  median kcal spread *within the correct food* is ~32% — `pinto bean` spans 82 kcal/100g canned
  to 333 dry, and the recipe line never says which. A perfect resolver is still that far off.
- **The machine learning earned nothing.** Eight engineered features, a calibrated logistic model
  and isotonic routing were +3.8pp on the tuning split and −3.9pp on the frozen holdout — the
  direction flipped, which is what noise does. What ships is the top embedding cosine plus a
  stored confidence curve.
- **Label quality is measured, not asserted.** Three independent LLM annotators over 120 strings
  give **Krippendorff's α = 0.709** — usable for directional claims, not precise ones. An
  ensemble could not improve the existing labels, so they were left in place.

> ⚠️ 282 of the 300 gold labels were produced by an LLM annotator rather than a human. Every
> number above measures agreement with those labels, not with ground truth. Read
> [`docs/er_metrics.md`](docs/er_metrics.md) before quoting any of them — it also records the
> measurements that failed, and why.

## Status

**Phase 1 complete** — Bronze landed and DuckDB-verified: `bronze.raw_recipes` (15,000
RecipeNLG rows, the subset-first sample) and `bronze.raw_usda_foods` (8,187 USDA Foundation +
SR Legacy foods).

**Phase 2 complete** (Silver + entity resolution — the headline): 112,463 ingredient lines parsed
to 9,324 distinct strings, 8,187 canonical USDA entities, a 300-row stratified gold set,
candidate generation at recall@50 = 90.5%, a shipped resolver, and the entity map above. 2.6's
Claude adjudicator was **investigated and not built** — its accuracy turns out to be structurally
unmeasurable against an LLM-labeled gold set (§10).

**Phase 3 next** — unit→gram conversion and servings, both of which need the FDC bulk download
for `foodPortions.gramWeight` (absent from the abridged API payload).

- Measured results, including the negative ones: [`docs/er_metrics.md`](docs/er_metrics.md)
- Labeling convention: [`docs/labeling_guide.md`](docs/labeling_guide.md)
- Session log + decisions: [`docs/session_2026-07-24.md`](docs/session_2026-07-24.md)

### Reproducing the entity resolution

```
uv run python -m pantryiq.silver.ingredient_lines   # -> silver.recipe_ingredient_lines
uv run python -m pantryiq.silver.usda_foods         # -> silver.usda_foods
uv run python -m pantryiq.er.candidates             # -> silver.ingredient_candidates
uv run python -m pantryiq.er.resolve                # fit the confidence curve
uv run python -m pantryiq.er.entity_map             # -> silver.ingredient_entity_map + headline
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
make setup   # create the venv (Python 3.11) and install dev deps
make test    # run tests
make lint    # ruff check
```

Copy `.env.example` to `.env` and fill in keys.

## Data

Datasets are gitignored (large). To reproduce Phase 1:

- **RecipeNLG** (~2.2M recipes, ~2.3 GB) — download `full_dataset.csv` from
  <https://recipenlg.cs.put.poznan.pl/> (accept the terms form), place it at
  `data/raw/recipenlg/full_dataset.csv`, then sanity-check:
  `uv run python scripts/inspect_recipenlg.py`.
- **USDA FoodData Central** — free API key from
  <https://fdc.nal.usda.gov/api-key-signup.html> → put in `.env` as `USDA_API_KEY`.

### Regenerating Bronze

```
uv run python -m pantryiq.ingestion.recipes   # -> bronze.raw_recipes    (15,000 rows)
uv run python -m pantryiq.ingestion.usda      # -> bronze.raw_usda_foods (8,187 rows)
```

Both are idempotent (each re-run overwrites its table; a 0-row result is refused rather than
silently wiping the table). The USDA counts are a **live** pull from 2026-07-22 — Foundation
grows over time, so a later pull may exceed 394 Foundation / 7,793 SR Legacy. The lakehouse
lives in `data/lakehouse/` (gitignored); delete that directory for a clean regeneration.
