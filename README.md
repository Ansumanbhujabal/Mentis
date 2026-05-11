# Mentis

Procurement intelligence reports for medical substances. Type a substance, get an 8-section, citation-grounded brief in < 90 seconds.

> POC in active development. See `docs/superpowers/specs/2026-05-11-mentis-poc-design.md` for the full design.

## Quickstart

```bash
uv sync
cp .env.example .env  # fill in keys
uv run mentis report "ranitidine" --out report.md --pdf report.pdf
# or:
uv run python app.py  # launches Gradio
```
