"""All Pydantic data models for Mentis."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

SectionName = Literal[
    "product_profile",
    "clinical_use",
    "market_demand",
    "manufacturers",
    "regulatory",
    "sourcing_pricing",
    "risks_alternatives",
]

SectionNames: tuple[SectionName, ...] = (
    "product_profile",
    "clinical_use",
    "market_demand",
    "manufacturers",
    "regulatory",
    "sourcing_pricing",
    "risks_alternatives",
)


class SectionPlan(BaseModel):
    section_name: SectionName
    search_queries: list[str]
    sources: list[str]


class QueryPlan(BaseModel):
    user_query: str
    normalized_term: str | None = None
    rxnav_synonyms: list[str] = Field(default_factory=list)
    section_plans: list[SectionPlan]
    plan_version: str = "v1"


class Snippet(BaseModel):
    text: str
    url: HttpUrl
    source_name: str
    source_kind: Literal["scientific", "regulatory", "market", "background"]
    title: str | None = None
    retrieved_at: datetime


class SectionDraft(BaseModel):
    section_name: SectionName
    prose: str
    snippets_used: list[Snippet]
    citations: list[HttpUrl]
    synthesizer_version: str
    safety_retries: int = 0
    fallback_to_raw_snippets: bool = False


class Reference(BaseModel):
    url: HttpUrl
    source_name: str
    title: str
    retrieved_at: datetime
    used_in_sections: list[str] = Field(default_factory=list)


class ReportMetadata(BaseModel):
    mentis_version: str
    llm_provider: str
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    total_latency_ms: int
    total_snippets_retrieved: int
    total_safety_retries: int = 0
    cost_usd: float = 0.0
    generated_at: datetime


class Report(BaseModel):
    user_query: str
    normalized_term: str | None = None
    sections: list[SectionDraft]
    executive_summary: str
    references: list[Reference]
    metadata: ReportMetadata
