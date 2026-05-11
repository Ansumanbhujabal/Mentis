"""PDF export via Markdown → HTML → WeasyPrint."""
from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

from mentis.report import SECTION_TITLES
from mentis.schemas import Report

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )


def _md_to_html(md_text: str) -> str:
    # gfm-like preset enables tables (commonmark profile doesn't).
    # Also enable linkify so bare URLs become hyperlinks.
    return MarkdownIt("gfm-like").enable(["linkify", "table"]).render(md_text)


# Branded CSS used for polished PDFs (the WeasyPrint stylesheet).
# Big upgrade over the inline stub: alternating row stripes, soft horizontal-only
# borders, sticky-looking header, generous column padding, anti-overflow word-break.
POLISHED_CSS = """
@page {
  size: A4;
  margin: 1.6cm 1.4cm 1.8cm 1.4cm;
  @bottom-center {
    content: "Mentis Procurement Brief  ·  Page " counter(page) " of " counter(pages);
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 8.5pt;
    color: #888;
  }
}
body {
  font-family: "Helvetica", "Arial", sans-serif;
  color: #1a1a1a;
  font-size: 10.5pt;
  line-height: 1.55;
}
h1 {
  font-size: 20pt;
  letter-spacing: 0.03em;
  color: #1a1a1a;
  border-bottom: 3px solid #1a1a1a;
  padding-bottom: 0.4em;
  margin-top: 0;
}
h2 {
  font-size: 13pt;
  color: #1a1a1a;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.3em;
  margin-top: 1.6em;
  margin-bottom: 0.6em;
  page-break-after: avoid;
}
h3 { font-size: 11pt; margin-top: 1em; margin-bottom: 0.4em; }
p { text-align: justify; margin: 0.6em 0; }
strong { color: #0a2540; }
em { color: #4a4a4a; }
a { color: #0a4d8c; text-decoration: none; word-break: break-word; }
a:hover { text-decoration: underline; }

/* Tables — the main visual upgrade */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.9em 0 1.2em 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
  border-top: 2px solid #1a1a1a;
  border-bottom: 2px solid #1a1a1a;
}
thead th {
  background: #f4f4f0;
  color: #1a1a1a;
  font-weight: 600;
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1.5px solid #1a1a1a;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  font-size: 8.5pt;
}
tbody td {
  padding: 7px 10px;
  vertical-align: top;
  border-bottom: 1px solid #e0e0e0;
  word-break: break-word;
  hyphens: auto;
}
tbody tr:nth-child(even) td { background: #fafafa; }
tbody tr:last-child td { border-bottom: none; }
tbody td a { color: #0a4d8c; }

/* Blockquotes (used for fallback notices) */
blockquote {
  margin: 0.8em 0;
  padding: 0.4em 0.9em;
  border-left: 3px solid #c08020;
  background: #fff8ec;
  color: #6a4a10;
  font-size: 9.5pt;
  font-style: italic;
}

/* Lists */
ul, ol { padding-left: 1.4em; margin: 0.6em 0; }
li { margin-bottom: 0.3em; }

/* References list — denser */
ol.references li, ol li[id^="ref"] { font-size: 9.5pt; line-height: 1.45; }

/* Page breaks */
h2 + table { margin-top: 0.5em; }
"""


def render_polished_html(polished_markdown: str) -> str:
    """Wrap polished markdown in a styled HTML document for WeasyPrint."""
    body_html = _md_to_html(polished_markdown)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{POLISHED_CSS}</style></head><body>"
        f"{body_html}"
        "</body></html>"
    )


def polished_markdown_to_pdf_bytes(polished_markdown: str) -> bytes | None:
    """Render polished markdown to PDF bytes via the upgraded stylesheet."""
    try:
        from weasyprint import HTML
    except Exception as e:
        logger.error(f"WeasyPrint import failed: {e!r}")
        return None
    try:
        return HTML(string=render_polished_html(polished_markdown), base_url=".").write_pdf()
    except Exception as e:
        logger.error(f"Polished PDF render failed: {e!r}")
        return None


def report_to_pdf_bytes(report: Report) -> bytes | None:
    """Render Report → PDF bytes. Returns None on failure."""
    try:
        from weasyprint import HTML
    except Exception as e:
        logger.error(f"WeasyPrint import failed: {e!r}")
        return None

    sections_html = [
        {
            "title": SECTION_TITLES.get(s.section_name, s.section_name),
            "html": _md_to_html(s.prose),
            "fallback": s.fallback_to_raw_snippets,
        }
        for s in report.sections
    ]
    executive_summary_html = _md_to_html(report.executive_summary)

    template = _env().get_template("report.html.j2")
    html_str = template.render(
        report=report,
        sections_html=sections_html,
        executive_summary_html=executive_summary_html,
    )

    try:
        return HTML(string=html_str, base_url=".").write_pdf()
    except Exception as e:
        logger.error(f"WeasyPrint PDF render failed: {e!r}")
        return None


def report_to_pdf_file(report: Report, out_path: Path) -> bool:
    pdf_bytes = report_to_pdf_bytes(report)
    if pdf_bytes is None:
        return False
    out_path.write_bytes(pdf_bytes)
    return True
