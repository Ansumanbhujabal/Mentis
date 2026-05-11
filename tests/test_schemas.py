"""Pydantic schemas validate correctly."""
from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from mentis.schemas import (
    QueryPlan,
    ReportMetadata,
    SectionDraft,
    SectionPlan,
    Snippet,
)


def test_query_plan_requires_exactly_seven_sections() -> None:
    sp = [
        SectionPlan(section_name="product_profile", search_queries=["q"], sources=["rxnav"])
        for _ in range(7)
    ]
    qp = QueryPlan(
        user_query="ranitidine",
        normalized_term="ranitidine hydrochloride",
        rxnav_synonyms=["Zantac"],
        section_plans=sp,
        plan_version="v1",
    )
    assert len(qp.section_plans) == 7


def test_section_name_is_enum() -> None:
    with pytest.raises(ValidationError):
        SectionPlan(section_name="not_a_real_section", search_queries=[], sources=[])


def test_snippet_has_source_kind() -> None:
    s = Snippet(
        text="A study found...",
        url="https://example.org/x",
        source_name="PubMed",
        source_kind="scientific",
        title=None,
        retrieved_at=datetime.now(),
    )
    assert s.source_kind == "scientific"


def test_section_draft_tracks_safety_retries() -> None:
    sd = SectionDraft(
        section_name="market_demand",
        prose="...",
        snippets_used=[],
        citations=[],
        synthesizer_version="v1",
        safety_retries=2,
        fallback_to_raw_snippets=False,
    )
    assert sd.safety_retries == 2


def test_report_metadata_includes_cost() -> None:
    m = ReportMetadata(
        mentis_version="0.1.0",
        llm_provider="gemini/gemini-2.0-flash",
        prompt_versions={"section_synthesizer": "v1"},
        total_latency_ms=80000,
        total_snippets_retrieved=47,
        total_safety_retries=0,
        cost_usd=0.0124,
        generated_at=datetime.now(),
    )
    assert m.cost_usd > 0
