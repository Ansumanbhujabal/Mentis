from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from mentis.llm import LLMConfig, SafetyTrace
from mentis.report import assemble_report, render_markdown
from mentis.schemas import (
    Reference,
    Report,
    ReportMetadata,
    SectionDraft,
    Snippet,
)


class _ESLLMOut(BaseModel):
    executive_summary: str


def _draft(name, urls) -> SectionDraft:
    snips = [
        Snippet(
            text="t",
            url=u,
            source_name="PubMed",
            source_kind="scientific",
            retrieved_at=datetime.now(),
        )
        for u in urls
    ]
    return SectionDraft(
        section_name=name,
        prose=f"prose for {name}",
        snippets_used=snips,
        citations=urls,
        synthesizer_version="v1",
    )


@pytest.mark.asyncio
async def test_assemble_report_generates_executive_summary(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "executive_summary.v1.j2").write_text("summary of {{ user_query }}")

    drafts = {
        "product_profile":     _draft("product_profile",    ["https://a.org/1"]),
        "clinical_use":        _draft("clinical_use",       ["https://a.org/2"]),
        "market_demand":       _draft("market_demand",      ["https://a.org/3"]),
        "manufacturers":       _draft("manufacturers",      ["https://a.org/4"]),
        "regulatory":          _draft("regulatory",         ["https://a.org/5"]),
        "sourcing_pricing":    _draft("sourcing_pricing",   ["https://a.org/6"]),
        "risks_alternatives":  _draft("risks_alternatives", ["https://a.org/7"]),
    }

    fake_llm = AsyncMock(
        return_value=(
            _ESLLMOut(executive_summary="ES text"),
            "gemini/gemini-2.0-flash",
            SafetyTrace(),
        )
    )

    report = await assemble_report(
        user_query="ranitidine",
        normalized_term="ranitidine HCl",
        section_drafts=drafts,
        llm_config=LLMConfig(),
        prompts_dir=prompts_dir,
        total_latency_ms=80_000,
        llm_complete=fake_llm,
    )
    assert isinstance(report, Report)
    assert len(report.sections) == 7
    assert report.executive_summary == "ES text"
    assert len(report.references) == 7  # one URL per section, all unique
    assert report.metadata.total_snippets_retrieved == 7


def test_render_markdown_includes_metadata_header_and_refs() -> None:
    drafts = [
        SectionDraft(
            section_name="product_profile",
            prose="A [fact](https://a.org/1).",
            snippets_used=[],
            citations=["https://a.org/1"],
            synthesizer_version="v1",
        )
    ]
    report = Report(
        user_query="x",
        normalized_term="X",
        sections=drafts,
        executive_summary="ES",
        references=[
            Reference(
                url="https://a.org/1",
                source_name="PubMed",
                title="t",
                retrieved_at=datetime.now(),
                used_in_sections=["product_profile"],
            )
        ],
        metadata=ReportMetadata(
            mentis_version="0.1.0",
            llm_provider="gemini/gemini-2.0-flash",
            prompt_versions={},
            total_latency_ms=1000,
            total_snippets_retrieved=1,
            total_safety_retries=0,
            cost_usd=0.001,
            generated_at=datetime.now(),
        ),
    )
    md = render_markdown(report)
    assert "# Mentis Procurement Brief" in md
    assert "Executive Summary" in md
    assert "References" in md
    assert "https://a.org/1" in md
