from __future__ import annotations

from datetime import datetime

from mentis.pdf import report_to_pdf_bytes
from mentis.schemas import (
    Reference,
    Report,
    ReportMetadata,
    SectionDraft,
)


def _minimal_report() -> Report:
    return Report(
        user_query="ranitidine",
        normalized_term="ranitidine HCl",
        sections=[
            SectionDraft(
                section_name="product_profile",
                prose="Ranitidine is an [H2 antagonist](https://a.org/1).",
                snippets_used=[],
                citations=["https://a.org/1"],
                synthesizer_version="v1",
            )
        ],
        executive_summary="A short executive summary.",
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


def test_report_to_pdf_bytes_returns_pdf() -> None:
    pdf_bytes = report_to_pdf_bytes(_minimal_report())
    assert pdf_bytes is not None
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 1000
