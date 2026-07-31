.PHONY: setup test lint gold gate prove-gate

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

# Build Gold WITH its tests interleaved. `dbt build` is deliberate: `dbt run` then `dbt test`
# would rebuild every Gold table from bad data and report the failure afterwards. build stops
# the DAG at the failure, so nothing downstream is written.
gate:
	uv run dbt build --profiles-dir . --target staging

# Gate, then swap into `gold` in one transaction and export the read-only Gold artifact.
gold:
	uv run python -m pantryiq.gold.publish

# Demonstrate that the gate actually blocks, rather than trusting that it would.
prove-gate:
	uv run python scripts/prove_gate.py
