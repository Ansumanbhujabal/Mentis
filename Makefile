.PHONY: install diagram sync-prompts deploy samples test lint

install:
	uv sync

diagram:
	uv run python docs/architecture.py

sync-prompts:
	uv run python scripts/sync_prompts.py

samples:
	uv run mentis report "0.9% saline" --out walkthrough/sample_reports/saline_0.9pct.md --pdf walkthrough/sample_reports/saline_0.9pct.pdf
	uv run mentis report "tramadol" --out walkthrough/sample_reports/tramadol.md --pdf walkthrough/sample_reports/tramadol.pdf
	uv run mentis report "insulin glargine" --out walkthrough/sample_reports/insulin_glargine.md --pdf walkthrough/sample_reports/insulin_glargine.pdf

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
