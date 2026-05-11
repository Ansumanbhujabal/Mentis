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
    return MarkdownIt("commonmark").enable("linkify").render(md_text)


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
