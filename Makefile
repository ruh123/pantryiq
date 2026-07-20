.PHONY: setup test lint

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
