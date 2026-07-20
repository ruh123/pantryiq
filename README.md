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

Datasets go under `data/raw/` (gitignored). Copy `.env.example` to `.env` and fill in keys.
