# PantryIQ

A verified recipe data platform with a data-grounded AI layer. The headline is **entity
resolution**: free-text recipe ingredients resolved to canonical USDA food entities with
measured precision/recall — so the AI layer can only speak from verified nutrition, cost, and
ingredient data, with a deterministic check on every numeric claim it makes.

- **Operating brief (locked decisions, phase gates):** [`docs/PantryIQ_Master_Prompt.md`](docs/PantryIQ_Master_Prompt.md)
- **Full design:** [`docs/PantryIQ_Design_Doc.md`](docs/PantryIQ_Design_Doc.md)

## Status

Phase 0 complete (scaffold + toolchain). Phase 1 in progress — RecipeNLG validated
(`scripts/inspect_recipenlg.py`); Bronze ingestion next.

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
